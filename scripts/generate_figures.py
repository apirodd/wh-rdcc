import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import trustworthiness
import numpy as np

FIGS_OUT = '/content/drive/MyDrive/WH_RDCC/figures'

# ── Figura 1: Curva λ vs metriche ─────────────────────────────
lambdas = [0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
maes  = [all_results_complete[l]['mae'] for l in lambdas]
tws   = [all_results_complete[l]['tw']  for l in lambdas]
cts   = [all_results_complete[l]['ct']  for l in lambdas]
pes   = [all_results_complete[l]['pe']  for l in lambdas]
srs   = [all_results_complete[l]['sr']  for l in lambdas]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.plot(lambdas, maes, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.axhline(maes[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax.scatter([5.0], [min(maes)], color='red', s=100, zorder=5, label=f'Opt λ=5')
ax.set_xlabel('λ'); ax.set_ylabel('MAE test [m]')
ax.set_title('Localizzazione vs λ'); ax.legend()
ax.set_xscale('symlog', linthresh=0.1)

ax = axes[1]
ax.plot(lambdas, tws, 's-', color='darkorange', linewidth=2,
        markersize=6, label='Trustworthiness')
ax.plot(lambdas, cts, '^-', color='green', linewidth=2,
        markersize=6, label='Continuity')
ax.axhline(tws[0], color='darkorange', linestyle='--', alpha=0.4)
ax.axhline(cts[0], color='green',      linestyle='--', alpha=0.4)
ax.set_xlabel('λ'); ax.set_ylabel('Score')
ax.set_title('Qualità geometrica vs λ')
ax.legend(); ax.set_xscale('symlog', linthresh=0.1)

ax = axes[2]
ax2 = ax.twinx()
ax.plot(lambdas, pes,  'd-', color='purple', linewidth=2,
        markersize=6, label='Pred Error')
ax2.plot(lambdas, srs, 'v--', color='brown', linewidth=2,
         markersize=6, label='sr(W*)', alpha=0.7)
ax.set_xlabel('λ'); ax.set_ylabel('Pred Error', color='purple')
ax2.set_ylabel('Spectral Radius', color='brown')
ax.set_title('Dinamica latente vs λ')
ax.set_xscale('symlog', linthresh=0.1)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2)

plt.suptitle('WH-RDCC: Ablation study su λ (DICHASUS cf02+cf03+cf04)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS_OUT}/fig1_lambda_ablation.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGS_OUT}/fig1_lambda_ablation.png', dpi=300, bbox_inches='tight')
plt.show()
print('Figura 1 salvata.')

# ── Figura 2: Channel Chart side-by-side ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

configs = [
    (z_train, p_train, 'Baseline (λ=0)\n'
     f'TW={all_results_complete[0]["tw"]:.3f}  '
     f'MAE={all_results_complete[0]["mae"]:.2f}m'),
    (z_tr5,   p_tr5,   'WH-RDCC λ=5\n'
     f'TW={all_results_complete[5.0]["tw"]:.3f}  '
     f'MAE={all_results_complete[5.0]["mae"]:.2f}m'),
    (z_tr20,  p_tr20,  'WH-RDCC λ=20\n'
     f'TW={all_results_complete[20.0]["tw"]:.3f}  '
     f'MAE={all_results_complete[20.0]["mae"]:.2f}m'),
]

for ax, (z, p, title) in zip(axes, configs):
    sc = ax.scatter(z[:,0], z[:,1], c=p[:,0],
                    cmap='RdYlGn', s=5, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='x [m]')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('z₁'); ax.set_ylabel('z₂')

plt.suptitle('Channel Chart: Baseline vs WH-RDCC',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS_OUT}/fig2_channel_charts.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGS_OUT}/fig2_channel_charts.png', dpi=300, bbox_inches='tight')
plt.show()
print('Figura 2 salvata.')

# ── Figura 3: Spettro W* per λ=0, 5, 20 ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

W_configs = [
    (W_b, 'W* Baseline (λ=0)',  all_results_complete[0]['sr']),
    (W5,  'W* WH-RDCC (λ=5)',   all_results_complete[5.0]['sr']),
    (W20, 'W* WH-RDCC (λ=20)',  all_results_complete[20.0]['sr']),
]

for ax, (W_plt, title, sr_val) in zip(axes, W_configs):
    eigvals = torch.linalg.eigvals(W_plt).cpu().numpy()
    mags_p  = np.abs(eigvals)

    circle = plt.Circle((0,0), 1, fill=False, color='gray',
                         linestyle='--', linewidth=1.5)
    ax.add_patch(circle)
    sc = ax.scatter(eigvals.real, eigvals.imag,
                    c=mags_p, cmap='plasma', s=100,
                    vmin=0.9, vmax=1.0, zorder=5)
    plt.colorbar(sc, ax=ax, label='|λ|')
    for i, (e, m) in enumerate(zip(eigvals, mags_p)):
        ax.annotate(f'λ{i+1}\n{m:.3f}', (e.real, e.imag),
                    fontsize=6, ha='center', va='bottom')
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3)
    ax.set_aspect('equal')
    ax.axhline(0,color='k',linewidth=0.5)
    ax.axvline(0,color='k',linewidth=0.5)
    ax.set_title(f'{title}\nsr={sr_val:.4f}  rank={int((mags_p>0.1).sum())}',
                 fontsize=10)
    ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')

plt.suptitle('Spettro di W*: interpretabilità cinematica',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS_OUT}/fig3_spectrum_W.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGS_OUT}/fig3_spectrum_W.png', dpi=300, bbox_inches='tight')
plt.show()
print('Figura 3 salvata.')

# ── Figura 4: Localizzazione test side-by-side ────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

loc_configs = [
    (p_test,  affine_align(z_train, p_train, z_test),
     f'Baseline  MAE={all_results_complete[0]["mae"]:.2f}m'),
    (p_te5,   p_pred5,
     f'WH-RDCC λ=5  MAE={all_results_complete[5.0]["mae"]:.2f}m'),
    (p_te20,  p_pred20,
     f'WH-RDCC λ=20  MAE={all_results_complete[20.0]["mae"]:.2f}m'),
]

for ax, (p_gt, p_pr, title) in zip(axes, loc_configs):
    ax.scatter(p_gt[:,0], p_gt[:,1],
               c='lightblue', s=8, label='Ground truth', alpha=0.7)
    ax.scatter(p_pr[:,0], p_pr[:,1],
               c='red', s=5, label='Predetto', alpha=0.5)
    # Linee di errore su subset
    for i in range(0, len(p_gt), 10):
        ax.plot([p_gt[i,0], p_pr[i,0]], [p_gt[i,1], p_pr[i,1]],
                'k-', alpha=0.15, linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(fontsize=7); ax.set_aspect('equal')

plt.suptitle('Localizzazione: Baseline vs WH-RDCC',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS_OUT}/fig4_localization.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGS_OUT}/fig4_localization.png', dpi=300, bbox_inches='tight')
plt.show()
print('Figura 4 salvata.')

print('\n=== TUTTE LE FIGURE SALVATE ===')
for f in sorted(os.listdir(FIGS_OUT)):
    kb = os.path.getsize(f'{FIGS_OUT}/{f}') / 1e3
    print(f'  {f:<40} {kb:.0f} KB')