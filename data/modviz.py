import os, joblib, pandas as pd, numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.align import Align
from rich.prompt import Prompt, Confirm, IntPrompt

try:
    from machine_learning import (
        Model, get_train_test_data, train_model_cv, evaluate_model_test, 
        get_model_config
    )
except ImportError:
    exit("Erreur: machine_learning.py introuvable.")

c = Console()
BASE_DIR = Path(__file__).resolve().parent
MOD_DIR = BASE_DIR / "mod"
DATA_DIR = BASE_DIR / "dataframe"
MOD_DIR.mkdir(exist_ok=True)

_data_cache = {}

def load_data():
    if _data_cache: return _data_cache['split']
    data_path = DATA_DIR / "data_clean.csv"
    if not data_path.exists():
        c.print(f"[red]Fichier {data_path} introuvable ![/]")
        exit()
    with c.status("[bold green]Chargement des données..."):
        df = pd.read_csv(data_path)
        splits = get_train_test_data(df)
        _data_cache['split'] = splits
    return splits

def get_existing_models_names():
    existing = set()
    if MOD_DIR.exists():
        for f in os.listdir(MOD_DIR):
            if f.startswith("res_") and f.endswith(".joblib"):
                name = f.replace("res_", "").replace(".joblib", "")
                existing.add(name.split('_')[0]) 
    return existing

def parse_input_list(value_str, type_cast):
    try:
        if not value_str.strip(): return None
        parts = [x.strip() for x in value_str.split(',')]
        if type_cast == 'int_list': return [int(x) for x in parts]
        elif type_cast == 'float_list': return [float(x) for x in parts]
        elif type_cast == 'str_list': return parts
    except: return None
    return None

def fmt(val):
    if isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    return str(val)

def display_full_report(model):
    c.clear()
    c.print(Panel(Align.center(f"[bold gold1]RAPPORT : {model.name.upper()}[/]"), style="gold1"))

    metrics_train = model.metrics.get('train', {})
    metrics_test = model.metrics.get('test', {})
    
    t_metrics = Table(title="Performances", box=box.SIMPLE_HEAVY, expand=True)
    t_metrics.add_column("Métrique", style="cyan")
    t_metrics.add_column("Train", style="green")
    t_metrics.add_column("Test", style="yellow")
    t_metrics.add_column("Écart", justify="right")

    keys = ['rmse', 'mae', 'mape', 'r2', 'median_ae']
    
    for k in keys:
        if k in metrics_train:
            val_tr = metrics_train[k]
            val_te = metrics_test.get(k, None)
            
            str_tr = fmt(val_tr)
            str_te = fmt(val_te) if val_te is not None else "-"
            
            str_delta = "-"
            if val_te is not None:
                delta = val_te - val_tr
                is_bad = (k != 'r2' and delta > 0.05) or (k == 'r2' and delta < -0.05)
                color = "red" if is_bad else "green"
                sign = "+" if delta > 0 else ""
                str_delta = f"[{color}]{sign}{delta:.4f}[/]"
            
            t_metrics.add_row(k.upper(), str_tr, str_te, str_delta)

    mean_cv = np.mean(model.cv_scores) if model.cv_scores else 0.0
    std_cv = np.std(model.cv_scores) if model.cv_scores else 0.0
    
    p_cv = Text()
    p_cv.append(f"Mean CV (MSE): ", style="bold")
    p_cv.append(f"{mean_cv:.4f}", style="green")
    p_cv.append(f" (± {std_cv:.4f})\n", style="dim")
    
    p_params = Text()
    if model.parameters:
        for k, v in model.parameters.items():
            short_k = k.split('__')[-1]
            p_params.append(f"• {short_k}: ", style="cyan")
            p_params.append(f"{v}\n", style="white")
    else:
        p_params.append("Paramètres par défaut", style="dim italic")

    grid = Table.grid(expand=True, padding=2)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(
        Panel(p_params, title="Hyperparamètres", border_style="blue"),
        Panel(p_cv, title="Cross-Validation", border_style="magenta")
    )

    c.print(t_metrics)
    c.print(grid)

    if hasattr(model, 'grid_history') and not model.grid_history.empty:
        c.print("\n[bold]Historique GridSearch (Top 5)[/]")
        disp_hist = model.grid_history.head(5)
        t_hist = Table(box=box.MINIMAL, show_edge=False)
        cols = [col for col in disp_hist.columns if 'param_' in col or 'mean_test' in col]
        for col in cols:
            clean_col = col.replace('param_reg__', '').replace('mean_test_score', 'Score')
            t_hist.add_column(clean_col, style="dim")
        
        for _, row in disp_hist.iterrows():
            vals = [fmt(row[c]) if isinstance(row[c], (float, int)) else str(row[c]) for c in cols]
            t_hist.add_row(*vals)
        c.print(t_hist)

def configure_hyperparams(algo_name, default_config):
    estimator, param_schema = default_config
    if not param_schema: return {}
    c.print(f"\n[bold cyan]Configuration {algo_name}[/]")
    if not Confirm.ask("Veux-tu tuner ce modèle ?", default=True): return {}
    grid_params = {}
    c.print("[dim]Valeurs séparées par des virgules (ex: 0.1, 1.0). Laisser vide pour sauter.[/]")
    for param_key, p_type in param_schema.items():
        short_name = param_key.split('__')[-1]
        val = Prompt.ask(f"Valeurs pour [yellow]{short_name}[/]")
        parsed = parse_input_list(val, p_type)
        if parsed: grid_params[param_key] = parsed
    return grid_params

def train_new_models_workflow():
    x_train, x_test, y_train, y_test = load_data()
    configs = get_model_config()
    session_models = []

    while True:
        c.clear()
        existing_names = get_existing_models_names()
        c.print(Panel("[bold yellow]Création de Modèle[/]", expand=False))
        
        algos = list(configs.keys())
        t_menu = Table(show_header=False, box=box.SIMPLE)
        
        for i, alg in enumerate(algos):
            status = "[red][EXISTANT][/]" if alg in existing_names else "[dim](Nouveau)[/]"
            t_menu.add_row(f"[{i+1}]", alg, status)
        c.print(t_menu)
        
        choice = IntPrompt.ask("Choisis un algo", choices=[str(i+1) for i in range(len(algos))])
        algo_name = algos[choice-1]
        
        if algo_name in existing_names:
            if not Confirm.ask(f"[red]Le modèle {algo_name} existe déjà. L'écraser ?[/]", default=False):
                continue
        
        params = configure_hyperparams(algo_name, configs[algo_name])
        
        with c.status(f"[bold red]Entraînement de {algo_name}...[/]"):
            try:
                name_uniq = algo_name 
                model = train_model_cv(name_uniq, configs[algo_name][0], params, x_train, y_train)
                session_models.append(model)
            except Exception as e:
                c.print(f"[red]Erreur: {e}[/]")
                input()
                continue

        display_full_report(model)
        if not Confirm.ask("\n[bold]Entraîner un autre modèle ?[/] (Non = Aller au classement)", default=False): break
            
    select_champion_workflow(session_models, x_test, y_test)

def select_champion_workflow(models, x_test, y_test):
    while True:
        c.clear()
        if not models:
            c.print("[red]Aucun modèle en session.[/]")
            return

        models.sort(key=lambda m: np.mean(m.cv_scores))
        t_lead = Table(title="Leaderboard Session (CV MSE)", box=box.ROUNDED)
        t_lead.add_column("#", style="cyan"); t_lead.add_column("Nom", style="bold yellow"); t_lead.add_column("Mean CV", style="green")

        for i, m in enumerate(models):
            t_lead.add_row(str(i+1), m.name, f"{np.mean(m.cv_scores):.4f}")
        c.print(t_lead)
        
        idx = IntPrompt.ask("Tester quel modèle ?", choices=[str(i+1) for i in range(len(models))]) - 1
        champion = models[idx]
        
        c.print(f"\n[bold]Testing {champion.name}...[/]")
        champion = evaluate_model_test(champion, x_test, y_test)
        
        display_full_report(champion)
        
        c.print("\n[1] Sauvegarder et Quitter\n[2] Re-tuner (Refaire)\n[3] Retour Leaderboard")
        d = Prompt.ask("Choix", choices=["1", "2", "3"])
        
        if d == "1":
            fname = f"res_{champion.name}.joblib"
            joblib.dump(champion, MOD_DIR / fname)
            c.print(f"[bold green]Sauvegardé : mod/{fname}[/]")
            break
        elif d == "2":
            algo_base = champion.name
            models.pop(idx)
            configs = get_model_config()
            params = configure_hyperparams(algo_base, configs[algo_base])
            x_train, _, y_train, _ = load_data()
            with c.status("Re-training..."):
                new_model = train_model_cv(algo_base, configs[algo_base][0], params, x_train, y_train)
            display_full_report(new_model)
            models.append(new_model)
            input("Retour leaderboard...")
        elif d == "3": continue

def view_existing_models():
    while True:
        if not MOD_DIR.exists(): c.print("[red]Dossier mod introuvable[/]"); break
        files = sorted([f for f in os.listdir(MOD_DIR) if f.endswith('.joblib')])
        if not files: c.print("[yellow]Aucun modèle.[/]"); break
        c.clear()
        t = Table(title="Modèles Sauvegardés", box=box.ROUNDED)
        t.add_column("#"); t.add_column("Fichier")
        for i, f in enumerate(files): t.add_row(str(i+1), f)
        c.print(t)
        ch = Prompt.ask("Choix (Q=Retour)", default="Q")
        if ch.lower() == 'q': return
        try:
            idx = int(ch)-1
            if 0 <= idx < len(files):
                m = joblib.load(MOD_DIR / files[idx])
                display_full_report(m)
                input("\nEntrée pour retour...")
        except: pass

while True:
    c.clear()
    c.print(Panel.fit("[bold magenta]ML STUDIO 2.0[/]", border_style="magenta"))
    c.print("[1]  Voir les modèles sauvegardés")
    c.print("[2]  Créer / Entraîner de nouveaux modèles")
    c.print("[Q] Quitter")
    choice = Prompt.ask("Choix", choices=["1", "2", "q", "Q"])
    if choice == "1": view_existing_models()
    elif choice == "2": train_new_models_workflow()
    else: break