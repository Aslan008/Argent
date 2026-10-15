"""
plugins/leeth_reload.py — /leeth-reload: hot-swap the Leeth runtime weights.

Reloads leeth_best.pt from src/leeth/data into the live runtime singleton
(no Argent restart needed). Use after training/finetuning or manual checkpoint swaps.
"""

from ui import console


def command_leeth_reload(*args):
    try:
        from src.leeth.runtime import leeth

        console.print("[cyan]Reloading Leeth runtime...[/cyan]")
        ok = leeth.reload(force_device="cuda")
        if ok:
            n_params = sum(p.numel() for p in leeth.model.parameters()) if leeth.model is not None else 0
            console.print(
                f"[green]Leeth reloaded[/green] device={leeth.device} "
                f"params={n_params:,} ({n_params/1e6:.0f}M)"
            )
        else:
            console.print(f"[red]Reload failed:[/red] {leeth._init_error}")
    except Exception as e:
        console.print(f"[red]Leeth reload error:[/red] {e}")