import os
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from train_main import BiGCNNet  
from get_dataset import load_dataset

# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "checkpoints/ba_shapes/best_bignn_ba_shapes_seed42_layer1.pt"

# Must match training (see train.py) :contentReference[oaicite:3]{index=3}
HIDDEN = 146
NUM_LAYERS = 1
GATE = None

# Explain target
NODE_ID = 0
LAYER_IDX = "last"  #last train layer


# ============================================================
# 1) Utilities: layer input, u extraction, B extraction, Q contraction
# ============================================================
@torch.no_grad()
def get_layer_input_x(model: nn.Module, x0: torch.Tensor, edge_index: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """
    Returns the input x_in to the layer_idx-th conv layer:
      - layer_idx=0 -> data.x
      - layer_idx=k -> output after 0..k-1 layers forward
    Note: dropout doesn't take effect under model.eval() (demo.py dropout controlled by training):contentReference[oaicite:4]{index=4}
    """
    x = x0
    for i in range(layer_idx):
        x = model.layers[i](x, edge_index)
    return x


def extract_B_factorized(conv) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the "implicit factorized form" of the third-order interaction tensor B:
      Your Bilinear implementation: h = (Wl x) ⊙ (Wr x)
      Equivalent to: B_k = w_l,k ⊗ w_r,k
    So instead of explicitly constructing B[d_out, d_in, d_in], just get (Wl, Wr).
    - Wl: [d_out, d_in]
    - Wr: [d_out, d_in]
    """
    Wl = conv.bilinear_tr.w_l
    Wr = conv.bilinear_tr.w_r
    return Wl, Wr

@torch.no_grad()
def build_B_sym(Wl: torch.Tensor, Wr: torch.Tensor) -> torch.Tensor:
    """
    Build symmetric bilinear tensor B_sym:
      B_k_sym = 0.5*(w_l,k ⊗ w_r,k + w_r,k ⊗ w_l,k)
    Shapes:
      Wl, Wr: [d_out, d_in]
      return: [d_out, d_in, d_in]
    """
    # outer products for each k: [d_out, d_in, d_in]
    B_lr = Wl[:, :, None] * Wr[:, None, :]   # w_l,k w_r,k^T
    B_rl = Wr[:, :, None] * Wl[:, None, :]   # w_r,k w_l,k^T
    B_sym = 0.5 * (B_lr + B_rl)
    return B_sym

@torch.no_grad()
def contract_u_out_B(u: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Q = u ·_out B = sum_k u_k * B_k
    u: [d_out]
    B: [d_out, d_in, d_in]
    return Q: [d_in, d_in]
    """
    return torch.einsum("k,kij->ij", u, B)

@torch.no_grad()
def compute_Q_j_to_i_aligned(
    u_node: torch.Tensor,   # [N, d_out]
    B: torch.Tensor,        # [d_out, d_in, d_in]
) -> torch.Tensor:
    """
    Returns:
        Q_node: [N, d_in, d_in]

        Q_node[j] = Q for node j → i
        Non-neighbors: Q = 0
    """
    Q_node = torch.einsum("nc,cij->nij", u_node, B)
    return Q_node

@torch.no_grad()
def compute_Q_all_aligned(
    u_node: torch.Tensor,   # [N, d_out]
    B: torch.Tensor,
) -> torch.Tensor:
    """
    Returns:
        Q_node: [N, d_in, d_in]
    """
    return torch.einsum("nc,cij->nij", u_node, B)

@torch.no_grad()
def get_a_ii_from_recorded_aij(conv, node_id: int) -> torch.Tensor:
    """
    Get self-loop a_ii from real propagation record (last_aij_edge_index, last_aij).
    """
    if conv.last_aij_edge_index is None or conv.last_aij is None:
        raise RuntimeError(
            "No recorded (edge_index, edge_weight) found. "
            "Make sure conv.record_aij=True and you already ran a forward pass."
        )

    ei = conv.last_aij_edge_index
    aij = conv.last_aij
    src, dst = ei[0], ei[1]

    self_mask = (src == node_id) & (dst == node_id)
    if self_mask.sum().item() == 0:
        raise RuntimeError(
            "No self-loop (i->i) found in recorded a_ij. "
            "This means the REAL propagated edge_index had no self-loop for this node at this layer."
        )
    return aij[self_mask][0]  # scalar

@torch.no_grad()
def get_all_aij_from_recorded(conv):
    """
    Returns the (edge_index, a_ij) actually used for propagate in forward pass.

    Returns
    -------
    edge_index : Tensor [2, E]
        Edges used for propagation (including self-loops, in normalized order)
    a_ij : Tensor [E] or None
        Propagation weights corresponding to each edge
    """
    if conv.last_aij_edge_index is None:
        raise RuntimeError(
            "No recorded edge_index found. "
            "Make sure conv.record_aij=True and a forward pass has been run."
        )

    return conv.last_aij_edge_index, conv.last_aij

@torch.no_grad()
def get_incoming_aij(
    conv,
    node_id: int,
    include_self: bool = True,
):
    """
    Get a_ij for all incoming edges j -> node_id.

    Parameters
    ----------
    node_id : int
        Center node i
    include_self : bool
        Whether to include self-loop (i -> i)

    Returns
    -------
    js : Tensor [K]
        All neighbor nodes j
    a_ij : Tensor [K] or None
        Corresponding propagation weights
    """
    if conv.last_aij_edge_index is None:
        raise RuntimeError(
            "No recorded (edge_index, edge_weight) found. "
            "Make sure conv.record_aij=True and forward already ran."
        )

    ei = conv.last_aij_edge_index
    aij = conv.last_aij

    src, dst = ei[0], ei[1]

    mask = (dst == node_id)
    if not include_self:
        mask = mask & (src != node_id)

    js = src[mask]
    a_ij = None if aij is None else aij[mask]

    return js, a_ij

@torch.no_grad()
def extract_u_self_aii_j_to_i_aligned(
    model: nn.Module,
    conv,
    node_id: int,
    class_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        u_node: [N, d_out]   aligned with node indices
        a_node: [N]          a_ji (0 for non-neighbors)
        w_c:    [d_out]
    """
    w_c = model.classifier_linear.weight[class_id].detach()  # [d_out]
    N = conv.num_nodes

    js, a_ji = get_incoming_aij(conv, node_id, include_self=True)

    # Initialize
    u_node = torch.zeros(N, w_c.numel(), device=w_c.device)
    a_node = torch.zeros(N, device=w_c.device)

    if a_ji is not None and js.numel() > 0:
        u_node[js] = a_ji.unsqueeze(1) * w_c.unsqueeze(0)
        a_node[js] = a_ji

    return u_node, a_node, w_c

@torch.no_grad()
def extract_u_all_aligned(
    model: nn.Module,
    conv,
    class_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        u_node: [N, d_out]   node-level aggregated u
        a_node: [N]          total a_ij for each node
        w_c:    [d_out]
    """
    w_c = model.classifier_linear.weight[class_id].detach()
    N = conv.num_nodes

    ei, aij = get_all_aij_from_recorded(conv)
    src = ei[0]

    u_node = torch.zeros(N, w_c.numel(), device=w_c.device)
    a_node = torch.zeros(N, device=w_c.device)

    # scatter sum over source node j
    u_node.index_add_(0, src, aij.unsqueeze(1) * w_c.unsqueeze(0))
    a_node.index_add_(0, src, aij)

    return u_node, a_node, w_c

@torch.no_grad()
def extract_u_self_aii(
    model: nn.Module,
    conv,
    node_id: int,
    class_id: int,
    mode: str,   # "aii" | "j->i" | "all"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract (u, aij, w_c) according to mode.

    mode:
      1) "aii"   : self-loop
      2) "j->i"  : all j -> i
      3) "all"   : all edges
    """

    # ---------- w_c ----------
    w_c = model.classifier_linear.weight[class_id].detach()  # [d_out]

    # ======================================================
    # mode 1: self-loop a_ii
    # ======================================================
    if mode == "aii":
        a_ii = get_a_ii_from_recorded_aij(conv, node_id)   # scalar
        u = a_ii * w_c                                     # [d_out]
        return u, a_ii, w_c

    # ======================================================
    # mode 2: neighbors j -> i
    # ======================================================
    elif mode == "j->i":
        js, a_ji = get_incoming_aij(conv, node_id, include_self=True)
        # js: [K], a_ji: [K]

        if a_ji is None:
            raise RuntimeError("a_ij is None, cannot construct u.")

        # u_j = a_ji * w_c
        u = a_ji.unsqueeze(1) * w_c.unsqueeze(0)   # [K, d_out]
        return u, a_ji, w_c

    # ======================================================
    # mode 3: all edges
    # ======================================================
    elif mode == "all":
        _, aij = get_all_aij_from_recorded(conv)
        # aij: [E]

        if aij is None:
            raise RuntimeError("a_ij is None, cannot construct u.")

        # u_e = a_ij * w_c
        u = aij.unsqueeze(1) * w_c.unsqueeze(0)    # [E, d_out]
        return u, aij, w_c

    else:
        raise ValueError(f"Unknown mode: {mode}")


@torch.no_grad()
def contract_u_B_to_Q(u: torch.Tensor, Wl: torch.Tensor, Wr: torch.Tensor) -> torch.Tensor:
    """
    Paper-aligned symmetric interaction:
      B_k^{sym} = 0.5*(w_l,k w_r,k^T + w_r,k w_l,k^T)
      Q = sum_k u_k B_k^{sym}
        = 0.5*(Wl^T diag(u) Wr + Wr^T diag(u) Wl)
    This Q is symmetric by construction.
    """
    Wl_u = Wl * u.view(-1, 1)   # diag(u) Wl
    Wr_u = Wr * u.view(-1, 1)   # diag(u) Wr
    Q = 0.5 * (Wl_u.T @ Wr + Wr_u.T @ Wl)

    # Optional: numerical stability (doesn't change theory)
    Q_ = 0.5 * (Q + Q.T)
    return Q



@torch.no_grad()
def eig_decompose(Q: torch.Tensor, topk: int = 10):
    """
    Symmetric matrix eigenvalue decomposition: Q = sum_k lambda_k v_k v_k^T
    Returns top-k sorted by |lambda| in descending order
    """
    eigvals, eigvecs = torch.linalg.eigh(Q)  # ascending
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    k = min(topk, eigvals.numel())
    return eigvals[:k], eigvecs[:, :k]


@torch.no_grad()
def eig_decompose_extremes(Q: torch.Tensor, k: int = 10):
    """
    Symmetric matrix eigenvalue decomposition:
      Q = sum_i lambda_i v_i v_i^T

    Returns:
      - top_k_pos:   (eigvals_pos, eigvecs_pos)   largest k eigenvalues (positive principal directions)
      - top_k_neg:   (eigvals_neg, eigvecs_neg)   smallest k eigenvalues (negative principal directions)
    """
    # eigh returns ascending: lambda_1 <= ... <= lambda_d
    eigvals_ori, eigvecs_ori = torch.linalg.eigh(Q)

    k = min(k, eigvals_ori.numel() // 2)

    # Largest k (positive)
    eigvals_pos = eigvals_ori[-k:]                 # [k]
    eigvecs_pos = eigvecs_ori[:, -k:]               # [d, k]

    # Smallest k (negative)
    eigvals_neg = eigvals_ori[:k]                  # [k]
    eigvecs_neg = eigvecs_ori[:, :k]                # [d, k]

    # For readability, sort positive ones in descending order
    idx_pos = torch.argsort(eigvals_pos, descending=True)
    eigvals_pos = eigvals_pos[idx_pos]
    eigvecs_pos = eigvecs_pos[:, idx_pos]

    # Negative usually goes from most negative to close to 0, can keep ascending
    return (eigvals_pos, eigvecs_pos), (eigvals_neg, eigvecs_neg), (eigvals_ori, eigvecs_ori)



@torch.no_grad()
def direction_contribution(x_i: torch.Tensor, eigvals: torch.Tensor, eigvecs: torch.Tensor):
    """
    Strictly decompose according to quadratic form:
      x^T Q x = sum_k lambda_k (v_k^T x)^2
    """
    proj = eigvecs.T @ x_i                # [k]
    contrib = eigvals * (proj ** 2)       # [k]
    return proj, contrib


# ============================================================
# 2) Neighbor contribution (still based on recorded REAL a_ij)
# ============================================================
@torch.no_grad()
def neighbor_contributions(conv, w_c: torch.Tensor, node_id: int, topk: int = 15):
    """
    Neighbor-level contribution:
      contrib(j->i) = a_ij * <w_c, h_j>

    Note about h_j here:
      Directly use conv.last_h (what you recorded in forward is bilinear+LN x):contentReference[oaicite:5]{index=5}
    """
    if conv.last_aij_edge_index is None or conv.last_h is None:
        raise RuntimeError("No recorded tensors found. Make sure conv.record_aij=True and forward already ran.")

    ei = conv.last_aij_edge_index
    aij = conv.last_aij
    h = conv.last_h

    src, dst = ei[0], ei[1]
    mask = (dst == node_id)

    neigh = src[mask]
    a_ij = None if aij is None else aij[mask]

    # <w_c, h_j>
    dot = (h[neigh] * w_c).sum(dim=-1)
    contrib = dot if a_ij is None else a_ij * dot

    k = min(topk, contrib.numel())
    order = torch.argsort(contrib, descending=True)[:k]

    rows = []
    for idx in order.tolist():
        rows.append({
            "j": int(neigh[idx].item()),
            "a_ij": None if a_ij is None else float(a_ij[idx].item()),
            "dot": float(dot[idx].item()),
            "contrib": float(contrib[idx].item()),
        })
    return rows





# ============================================================
# (NEW) Edge-wise interaction: build B_n^sym and Q_n from (W, V, r)  Node-pair interaction strength B and Q
# ============================================================
@torch.no_grad()
def build_Bn_sym_and_Qn(conv):
    """
    Construct node-pair interaction tensor B_n (symmetrized version) and interaction matrix Q_n from dynamic edge weight module parameters:

      z_ij = (W h_i) ⊙ (V h_j)
      alpha_ij = r^T z_ij

    Where W = edge_proj_src.weight, V = edge_proj_dst.weight, r = rel_vec.

    1) B_n raw:   (B_n)_k = w_k ⊗ v_k
    2) B_n sym:   (B_n^sym)_k = 0.5*(w_k v_k^T + v_k w_k^T)
    3) Q_n:       Q_n = r ·_out B_n^sym = sum_k r_k (B_n^sym)_k   (symmetric matrix)
    """
    # W,V: [d_r, d_h]; r: [d_r]
    W = conv.edge_proj_src.weight.detach()
    V = conv.edge_proj_dst.weight.detach()
    r = conv.rel_vec.detach()

    # --- B_n^sym: [d_r, d_h, d_h]
    B_wv = W[:, :, None] * V[:, None, :]   # w_k v_k^T
    B_vw = V[:, :, None] * W[:, None, :]   # v_k w_k^T
    Bn_sym = 0.5 * (B_wv + B_vw)

    # --- Q_n = sum_k r_k * Bn_sym[k]
    Qn = torch.einsum("k,kij->ij", r, Bn_sym)
    Qn = 0.5 * (Qn + Qn.T)  # Numerical stability: ensure strict symmetry

    return Bn_sym, Qn


# ============================================================
# (NEW) Fetch alpha_ij for all neighbors j -> center (reuse trained model cache)
# ============================================================
@torch.no_grad()
def get_incoming_alpha(conv, center_id: int):
    """
    Return (j_list, alpha_list) for all incoming edges j -> center.
    Directly reuse cached conv.alpha from training/forward, don't recalculate.
    """
    if conv.last_aij_edge_index is None:
        raise RuntimeError("conv.last_aij_edge_index is None. Ensure conv.record_aij=True and a forward pass ran.")
    if not hasattr(conv, "alpha") or conv.alpha is None:
        raise RuntimeError("conv.alpha is None. Ensure use_dynamic_edge_weight=True and alpha is recorded in forward.")

    ei = conv.last_aij_edge_index
    src, dst = ei[0], ei[1]

    mask = (dst == center_id) & (src != dst)  # Incoming edges excluding self-loops (alpha also masked self-loops originally)
    js = src[mask]
    al = conv.alpha.detach()[mask]
    return js, al



@torch.no_grad()
def pair_feature_interaction_matrix(
    Qn: torch.Tensor,
    h_i: torch.Tensor,
    h_j: torch.Tensor,
) -> torch.Tensor:
    """
    Return the "feature pair interaction contribution matrix" C for node pair (i, j) in hidden feature dimensions:

        C[p, q] = h_i[p] * Qn[p, q] * h_j[q]

    Where:
        C.sum() == h_i^T Qn h_j   (within numerical error)

    Args:
        Qn: [d, d]  node-pair interaction matrix
        h_i: [d]    hidden representation of node i (recommended to use conv.last_h[i])
        h_j: [d]    hidden representation of node j (recommended to use conv.last_h[j])

    Returns:
        C: [d, d]   feature pair interaction contribution matrix
    """
    h_i = h_i.view(-1)
    h_j = h_j.view(-1)
    # C[p,q] = h_i[p] * Qn[p,q] * h_j[q]
    C = (h_i[:, None] * Qn) * h_j[None, :]
    return C


@torch.no_grad()
def topk_pairs_from_C(C: torch.Tensor, k: int = 20):
    """
    Take the top-k feature pairs with largest |C[p,q]| from contribution matrix C.
    Returns: list of (p, q, contrib)
    """
    flat = C.reshape(-1)
    k = min(k, flat.numel())
    idx = torch.argsort(flat, descending=True)[:k]
    d = C.size(0)

    pairs = []
    for t in idx.tolist():
        p = t // d
        q = t % d
        pairs.append((int(p), int(q), float(C[p, q].item())))
    return pairs








# ============================================================
# 3) Main
# ============================================================
def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    # ---- load data (same as training) ---- :contentReference[oaicite:6]{index=6}
    data = load_dataset("ba_shapes")
    data = data.to(DEVICE)

    # ---- build model (same as training hyperparams) ---- :contentReference[oaicite:7]{index=7}
    model = BiGCNNet(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN,
        out_channels=data.num_classes,
        num_layers=NUM_LAYERS,
        num_nodes=data.num_nodes,
        gate=GATE
    ).to(DEVICE)

    # ---- load checkpoint ---- :contentReference[oaicite:8]{index=8}
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}, best_test_acc={ckpt.get('best_test_acc','?')}")

    # ---- enable recording of REAL propagated a_ij and h ---- :contentReference[oaicite:9]{index=9}
    for conv in model.layers:
        conv.record_aij = True

    # ---- one forward pass to populate last_* in each layer ----
    with torch.no_grad():
        out = model(data.x, data.edge_index)     # log-prob
        pred = out.argmax(dim=-1)

    # ---- choose layer = last ----
    layer_idx = len(model.layers) - 1 if LAYER_IDX == "last" else int(LAYER_IDX)
    conv = model.layers[layer_idx]

    # ---- pick target class = predicted class of NODE_ID ----
    class_id = int(pred[NODE_ID].item())
    print(f"\n[Explain] layer_idx={layer_idx} (last), node_id={NODE_ID}, class_id(pred)={class_id}")

    # ========================================================
    # A) First extract u and B
    # ========================================================
    # u_{ii,c} = a_ii * w_c
    u, a_ii, w_c = extract_u_self_aii(model, conv, NODE_ID, class_id, mode="aii")   #mode="aii","j->i","all"

    # Factorized representation of B: Wl, Wr
    Wl, Wr = extract_B_factorized(conv)

    print("\n[Core objects]")
    print(f"  a_ii (real propagated) = {float(a_ii.item()):.6e}")
    print(f"  u shape = {tuple(u.shape)} (should be [d_out])")
    print(f"  Wl shape = {tuple(Wl.shape)}, Wr shape = {tuple(Wr.shape)} (both should be [d_out, d_in])")

    # ========================================================
    # B) Get Q from u and B, then perform eigenvalue decomposition
    # ========================================================
    B = build_B_sym(Wl, Wr)  # [d_out, d_in, d_in]
    Q = contract_u_out_B(u, B)  # [d_in, d_in]
    print(f"Q shape: {Q.shape}")
    print(f"Q matrix:\n{Q}")

    # last layer input x_in (for x^T Q x interpretation)
    x_in = get_layer_input_x(model, data.x, data.edge_index, layer_idx=layer_idx)
    x_i = x_in[NODE_ID]  # [d_in]

    # eigvals, eigvecs = eig_decompose(Q, topk=10)
    pos_k_eig, neg_k_eig, _ = eig_decompose_extremes(Q, k=10)      # Get top k and bottom k eigenvalues and their corresponding eigenvectors








    # ========================================================
    # (NEW) C) Node-pair interaction strength: construct Q_n + read all neighbor alpha_ij
    # ========================================================
    Bn_sym, Qn = build_Bn_sym_and_Qn(conv)  # B_n^sym: [d_r,d_h,d_h], Q_n: [d_h,d_h]

    js, alpha_in = get_incoming_alpha(conv, NODE_ID)  # all j -> NODE_ID alpha_ij

    print("\n[Edge interaction] (node-pair) reuse trained alpha_ij")
    print(f"  #incoming edges = {js.numel()}")

    # Show ordered by alpha (you want "strong interaction edges")
    if js.numel() > 0:
        order = torch.argsort(alpha_in, descending=True)
        js = js[order]
        alpha_in = alpha_in[order]

        top_show = min(20, js.numel())
        for t in range(top_show):
            print(f"  {t+1:02d}. j={int(js[t])} -> i={NODE_ID}   alpha_ij={float(alpha_in[t]): .6e}")

    i = NODE_ID
    j = int(js[0])  # e.g. some neighbor
    C = pair_feature_interaction_matrix(Qn, conv.last_h[i], conv.last_h[j])

    print(f"Heatmap between node {i} and node {j}:", C)
    # # Verify: sum(C) = h_i^T Qn h_j
    # score = float((conv.last_h[i] @ (Qn @ conv.last_h[j])).item())
    # print("sum(C) =", float(C.sum().item()), "score =", score)
    #
    # # Optional: take top-k feature pairs
    # pairs = topk_pairs_from_C(C, k=10)
    # print(pairs[:5])





if __name__ == "__main__":
    main()
