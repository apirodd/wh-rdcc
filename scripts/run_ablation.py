# Ablation su lambda — il parametro più critico
# Esegui training con lambda = 0.5, 1.0, 5.0 e confronta

LAMBDAS = [0.5, 1.0, 5.0]
results = {}

for lam in LAMBDAS:
    print(f'\n=== λ={lam} ===')
    enc_abl  = CSIEncoder(LATENT_DIM).to(device)
    res_abl  = ResidualNet(LATENT_DIM).to(device)
    opt_abl  = torch.optim.Adam(
        list(enc_abl.parameters()) + list(res_abl.parameters()),
        lr=1e-3, weight_decay=1e-4)
    sch_abl  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_abl, T_max=140, eta_min=1e-5)

    W_abl = torch.eye(LATENT_DIM, device=device)

    # Bootstrap 40 ep
    for ep in range(1, 41):
        enc_abl.train()
        for csi_b, pos_b in train_loader:
            B,L,a,s,c = csi_b.shape
            z = enc_abl(csi_b.view(B*L,a,s,c).to(device)).view(B,L,-1)
            l = contrastive_loss(z, pos_b.to(device))
            opt_abl.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(enc_abl.parameters(), 1.0)
            opt_abl.step()
        sch_abl.step()

    # Stima W*
    with torch.no_grad():
        all_z = extract_all_z_seq(enc_abl, train_loader, device)
        W_abl = estimate_wiener_hopf(all_z)

    # Joint 100 ep
    for ep in range(1, 101):
        enc_abl.train()
        for csi_b, pos_b in train_loader:
            B,L,a,s,c = csi_b.shape
            z_flat = enc_abl(csi_b.view(B*L,a,s,c).to(device))
            z_seq  = z_flat.view(B,L,-1)
            l_cc   = contrastive_loss(z_seq, pos_b.to(device))
            z_t    = z_seq[:,:-1,:]; z_tp1 = z_seq[:,1:,:]
            l_wh   = (z_tp1 - z_t @ W_abl.T).pow(2).mean()
            loss   = l_cc + lam * l_wh
            opt_abl.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_abl.parameters(), 1.0)
            opt_abl.step()
        sch_abl.step()
        if ep % WH_UPDATE_EVERY == 0:
            with torch.no_grad():
                all_z = extract_all_z_seq(enc_abl, train_loader, device)
                W_abl = estimate_wiener_hopf(all_z)

    # Valuta
    with torch.no_grad():
        z_tr, p_tr = extract_latents(enc_abl, train_loader, device)
        z_te, p_te = extract_latents(enc_abl, test_loader,  device)
    p_pred = affine_align(z_tr, p_tr, z_te)
    mae    = np.mean(np.linalg.norm(p_pred - p_te, axis=1))
    tw_abl = trustworthiness(p_tr, z_tr, n_neighbors=10)
    ct_abl = trustworthiness(z_tr, p_tr, n_neighbors=10)
    with torch.no_grad():
        all_z = extract_all_z_seq(enc_abl, train_loader, device)
        z_t   = all_z[:,:-1,:]; z_tp1 = all_z[:,1:,:]
        pe    = (z_tp1 - z_t @ W_abl.T).pow(2).sum(-1).sqrt().mean().item()
    sr = torch.abs(torch.linalg.eigvals(W_abl)).max().item()

    results[lam] = {'mae': mae, 'tw': tw_abl, 'ct': ct_abl, 'pe': pe, 'sr': sr}
    print(f'  MAE={mae:.3f}m  TW={tw_abl:.4f}  CT={ct_abl:.4f}  '
          f'PE={pe:.4f}  sr={sr:.4f}')

# Tabella finale completa
print('\n=== ABLATION λ ===')
print(f'{"λ":<8} {"MAE[m]":>8} {"TW":>8} {"CT":>8} {"PredErr":>10} {"sr(W*)":>8}')
print('-' * 52)
print(f'{"0 (base)":<8} {mae_test_b:>8.3f} {tw_b:>8.4f} {ct_b:>8.4f} '
      f'{pred_err_b:>10.4f} {sr_b:>8.4f}')
print(f'{"0.1":<8} {mae_test:>8.3f} {tw:>8.4f} {ct:>8.4f} '
      f'{pred_err_mean:>10.4f} {mags.max():>8.4f}')
for lam, r in results.items():
    print(f'{lam:<8} {r["mae"]:>8.3f} {r["tw"]:>8.4f} {r["ct"]:>8.4f} '
          f'{r["pe"]:>10.4f} {r["sr"]:>8.4f}')