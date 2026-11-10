#!/usr/bin/env python3
"""
Linha & Ritmo
Ferramenta computacional para análise e balanceamento de linhas de produção
por meio de simulação de eventos discretos.

Uso:
    python linha_ritmo.py

Módulos principais:
    - Station / LineConfig .... representação dos dados da linha
    - calc_indicators() ....... indicadores analíticos (takt time, tempo de
                                 ciclo, capacidade, eficiência, gargalo)
    - simulate() ............... simulação de eventos discretos (fluxo peça a
                                 peça pelos postos, com variabilidade)
    - Scenario ................. cenários alternativos de balanceamento
    - export_charts() .......... exporta gráficos (matplotlib) para o TCC
    - CLI (main) ............... menu interativo de linha de comando
"""

from __future__ import annotations

import os
import random
import copy
import webbrowser
from dataclasses import dataclass, field
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Modelos de dados
# --------------------------------------------------------------------------- #

@dataclass
class Station:
    name: str
    time: float  # tempo de operação, em segundos


@dataclass
class LineConfig:
    jornada: float          # horas por dia
    demanda: float          # peças por dia
    variabilidade: float    # % de variabilidade dos tempos (coef. de variação)
    npecas: int             # número de peças a simular
    stations: List[Station] = field(default_factory=list)


@dataclass
class Indicators:
    takt_time: float
    cycle_time: float
    sum_times: float
    n_stations: int
    capacidade: int
    eficiencia: float
    bottleneck_idx: int
    atende_demanda: bool


@dataclass
class SimResult:
    n: int
    makespan: float
    utilization: List[float]
    avg_wait: List[float]
    avg_lead_time: float
    throughput_hora: float
    bottleneck_sim_idx: int


@dataclass
class Scenario:
    name: str
    config: LineConfig
    indicators: Indicators
    sim: SimResult


# --------------------------------------------------------------------------- #
# Núcleo de cálculo
# --------------------------------------------------------------------------- #

def calc_indicators(cfg: LineConfig) -> Indicators:
    """Indicadores analíticos de balanceamento de linha."""
    times = [s.time for s in cfg.stations]
    cycle_time = max(times)
    sum_times = sum(times)
    n_stations = len(cfg.stations)

    takt_time = (cfg.jornada * 3600) / cfg.demanda
    capacidade = int((cfg.jornada * 3600) // cycle_time)
    eficiencia = (sum_times / (n_stations * cycle_time)) * 100
    bottleneck_idx = times.index(cycle_time)
    atende_demanda = cycle_time <= takt_time

    return Indicators(
        takt_time=takt_time,
        cycle_time=cycle_time,
        sum_times=sum_times,
        n_stations=n_stations,
        capacidade=capacidade,
        eficiencia=eficiencia,
        bottleneck_idx=bottleneck_idx,
        atende_demanda=atende_demanda,
    )


def simulate(cfg: LineConfig, seed: Optional[int] = None) -> SimResult:
    """
    Simulação de eventos discretos de uma linha de fluxo (flow line) com
    liberação de peças ritmada pelo takt time (puxada pela demanda).

    Peças são liberadas na entrada da linha em intervalos iguais ao takt
    time — refletindo a lógica de produção puxada, em que o ritmo de entrada
    de matéria-prima acompanha a demanda do cliente, e não a velocidade
    máxima do primeiro posto. Isso evita filas artificialmente crescentes
    quando a linha está bem balanceada e, ao mesmo tempo, evidencia o
    crescimento real da fila quando o tempo de ciclo (gargalo) ultrapassa o
    takt time — ou seja, quando a linha não acompanha o ritmo de demanda.

    Cada peça i só pode iniciar o posto j quando:
      (a) chegou à entrada da linha / terminou o posto j-1, e
      (b) o posto j está livre da peça anterior (i-1).
    O tempo de operação de cada (peça, posto) varia aleatoriamente em torno
    da média informada, conforme o percentual de variabilidade.
    """
    if seed is not None:
        random.seed(seed)

    m = len(cfg.stations)
    n = max(10, round(cfg.npecas))
    variability = max(0.0, cfg.variabilidade) / 100
    takt_time = (cfg.jornada * 3600) / cfg.demanda

    arrival = [i * takt_time for i in range(n)]  # liberação puxada pela demanda
    finish = [[0.0] * m for _ in range(n)]
    waiting = [[0.0] * m for _ in range(n)]
    station_busy = [0.0] * m

    for i in range(n):
        for j in range(m):
            base = cfg.stations[j].time
            svc = base * (1 + variability * random.gauss(0, 1))
            svc = max(base * 0.15, svc)  # tempo nunca cai a zero/negativo

            prev_process = finish[i][j - 1] if j > 0 else arrival[i]
            prev_free = finish[i - 1][j] if i > 0 else 0.0
            start = max(prev_process, prev_free)

            finish[i][j] = start + svc
            waiting[i][j] = start - prev_process
            station_busy[j] += svc

    makespan = finish[n - 1][m - 1]
    utilization = [(b / makespan) * 100 for b in station_busy]

    avg_wait = [sum(waiting[i][j] for i in range(n)) / n for j in range(m)]

    lead_sum = 0.0
    for i in range(n):
        lead_sum += finish[i][m - 1] - arrival[i]
    avg_lead_time = lead_sum / n

    throughput_hora = (n / makespan) * 3600
    bottleneck_sim_idx = utilization.index(max(utilization))

    return SimResult(
        n=n,
        makespan=makespan,
        utilization=utilization,
        avg_wait=avg_wait,
        avg_lead_time=avg_lead_time,
        throughput_hora=throughput_hora,
        bottleneck_sim_idx=bottleneck_sim_idx,
    )


def build_scenario(name: str, cfg: LineConfig) -> Scenario:
    indicators = calc_indicators(cfg)
    sim = simulate(cfg)
    return Scenario(name=name, config=cfg, indicators=indicators, sim=sim)


# --------------------------------------------------------------------------- #
# Exibição em texto (barras ASCII) — funciona sem dependências externas
# --------------------------------------------------------------------------- #

def _bar(value: float, max_value: float, width: int = 30) -> str:
    if max_value <= 0:
        filled = 0
    else:
        filled = int(round((value / max_value) * width))
        filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def print_indicators(name: str, cfg: LineConfig, ind: Indicators) -> None:
    print(f"\n=== {name} — indicadores ===")
    print(f"Takt time ......... {ind.takt_time:8.1f} s/peça")
    flag = "OK" if ind.atende_demanda else "NÃO atende à demanda"
    print(f"Tempo de ciclo .... {ind.cycle_time:8.1f} s   [{flag}]")
    print(f"Capacidade teórica  {ind.capacidade:8d} peças/dia")
    print(f"Eficiência ........ {ind.eficiencia:8.1f} %")
    print(f"Gargalo ........... {cfg.stations[ind.bottleneck_idx].name}")

    print("\nTempo por posto (barra cheia = maior tempo da linha):")
    max_t = max(s.time for s in cfg.stations)
    for i, s in enumerate(cfg.stations):
        mark = " <- gargalo" if i == ind.bottleneck_idx else ""
        print(f"  {s.name:<16} {_bar(s.time, max_t)} {s.time:6.1f}s{mark}")


def print_sim(sim: SimResult, cfg: LineConfig) -> None:
    print(f"\n=== Simulação de eventos discretos ({sim.n} peças) ===")
    print(f"Throughput simulado  {sim.throughput_hora:8.1f} peças/h")
    print(f"Lead time médio .... {sim.avg_lead_time:8.1f} s/peça")

    print("\nUtilização por posto:")
    for i, s in enumerate(cfg.stations):
        print(f"  {s.name:<16} {_bar(sim.utilization[i], 100)} {sim.utilization[i]:5.1f}%")

    print("\nFila média (espera antes do posto):")
    max_w = max(sim.avg_wait) or 1
    for i, s in enumerate(cfg.stations):
        print(f"  {s.name:<16} {_bar(sim.avg_wait[i], max_w)} {sim.avg_wait[i]:6.1f}s")


def print_comparison(baseline: Scenario, scenarios: List[Scenario]) -> None:
    all_scen = [baseline] + scenarios
    print("\n=== Comparação de cenários ===")
    header = f"{'Cenário':<20}{'Takt':>8}{'Ciclo':>8}{'Efic.%':>9}{'Cap/dia':>10}{'Pçs/h':>9}{'Lead(s)':>10}  Gargalo"
    print(header)
    print("-" * len(header))
    for sc in all_scen:
        gargalo = sc.config.stations[sc.indicators.bottleneck_idx].name
        print(
            f"{sc.name:<20}"
            f"{sc.indicators.takt_time:8.0f}"
            f"{sc.indicators.cycle_time:8.0f}"
            f"{sc.indicators.eficiencia:9.1f}"
            f"{sc.indicators.capacidade:10d}"
            f"{sc.sim.throughput_hora:9.1f}"
            f"{sc.sim.avg_lead_time:10.1f}  {gargalo}"
        )


# --------------------------------------------------------------------------- #
# Exportação de gráficos (opcional — requer matplotlib)
# --------------------------------------------------------------------------- #

def export_charts(baseline: Scenario, scenarios: List[Scenario], outdir: str = ".") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib não instalado — pulei a exportação de gráficos.")
        print("Instale com: pip install matplotlib")
        return

    import os
    os.makedirs(outdir, exist_ok=True)

    cfg, ind, sim = baseline.config, baseline.indicators, baseline.sim
    names = [s.name for s in cfg.stations]
    times = [s.time for s in cfg.stations]

    # 1) tempo por posto vs takt time
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#B8501E" if i == ind.bottleneck_idx else "#1F6F6B" for i in range(len(names))]
    ax.bar(names, times, color=colors)
    ax.axhline(ind.takt_time, color="black", linestyle="--", linewidth=1, label="Takt time")
    ax.set_ylabel("Tempo (s)")
    ax.set_title("Tempo de operação por posto — situação atual")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "01_tempos_por_posto.png"), dpi=150)
    plt.close(fig)

    # 2) utilização por posto
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, sim.utilization, color="#1F6F6B")
    ax.set_ylabel("Utilização (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Utilização dos postos (simulação)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "02_utilizacao_postos.png"), dpi=150)
    plt.close(fig)

    # 3) comparação entre cenários (se houver)
    all_scen = [baseline] + scenarios
    if len(all_scen) > 1:
        labels = [s.name for s in all_scen]
        eff = [s.indicators.eficiencia for s in all_scen]
        thr = [s.sim.throughput_hora for s in all_scen]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, eff, color="#1F6F6B")
        ax.set_ylabel("Eficiência (%)")
        ax.set_title("Eficiência de balanceamento por cenário")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "03_comparacao_eficiencia.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, thr, color="#1F6F6B")
        ax.set_ylabel("Peças / hora")
        ax.set_title("Throughput simulado por cenário")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "04_comparacao_throughput.png"), dpi=150)
        plt.close(fig)

    print(f"Gráficos exportados para: {os.path.abspath(outdir)}")


# --------------------------------------------------------------------------- #
# Interface gráfica (abre a versão web no navegador)
# --------------------------------------------------------------------------- #

def abrir_frontend() -> None:
    """Abre a interface gráfica (linha_ritmo.html) no navegador padrão.

    O arquivo HTML roda inteiramente no navegador (sem precisar de servidor
    nem de internet) e precisa estar salvo na mesma pasta deste script.
    """
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linha_ritmo.html")
    if not os.path.exists(html_path):
        print(f"\n[!] Não encontrei 'linha_ritmo.html' em: {os.path.dirname(html_path)}")
        print("    Baixe esse arquivo e coloque na mesma pasta de linha_ritmo.py, depois tente de novo.")
        return
    webbrowser.open("file://" + html_path)
    print(f"\nAbrindo a interface gráfica no navegador...")
    print(f"Se não abrir sozinho, cole este endereço no navegador:\n  file://{html_path}")


# --------------------------------------------------------------------------- #
# Entrada de dados (CLI)
# --------------------------------------------------------------------------- #

def _ask_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        print("Valor inválido, usando padrão.")
        return default


def _ask_positive(prompt: str, default: float) -> float:
    """Pede um valor numérico maior que zero (jornada, demanda e tempos)."""
    while True:
        value = _ask_float(prompt, default)
        if value > 0:
            return value
        print("O valor deve ser maior que zero.")


def _ask_int(prompt: str, default: int) -> int:
    return int(_ask_float(prompt, default))


def input_line_config(default_cfg: Optional[LineConfig] = None) -> LineConfig:
    print("\n--- Dados da linha de produção ---")
    jornada = _ask_positive("Jornada de trabalho (h/dia)", default_cfg.jornada if default_cfg else 8)
    demanda = _ask_positive("Demanda (peças/dia)", default_cfg.demanda if default_cfg else 360)
    variabilidade = _ask_float("Variabilidade dos tempos (%)", default_cfg.variabilidade if default_cfg else 10)
    npecas = _ask_int("Peças a simular", default_cfg.npecas if default_cfg else 250)

    stations: List[Station] = []
    if default_cfg and default_cfg.stations:
        print("\nPostos atuais (pressione Enter para manter, ou digite novo tempo em segundos):")
        for s in default_cfg.stations:
            t = _ask_positive(f"  {s.name}", s.time)
            stations.append(Station(name=s.name, time=t))
        if input("\nAdicionar mais postos? (s/N): ").strip().lower() == "s":
            stations.extend(_input_new_stations(len(stations)))
    else:
        print("\nCadastro dos postos de trabalho (digite um nome vazio para encerrar):")
        i = 1
        while True:
            name = input(f"  Nome do posto {i} (Enter para encerrar): ").strip()
            if not name:
                if i == 1:
                    print("É necessário ao menos um posto. Usando dados de exemplo.")
                    stations = _default_stations()
                break
            t = _ask_positive(f"  Tempo de operação de '{name}' (s)", 40)
            stations.append(Station(name=name, time=t))
            i += 1

    return LineConfig(jornada=jornada, demanda=demanda, variabilidade=variabilidade,
                       npecas=npecas, stations=stations)


def _input_new_stations(start_index: int) -> List[Station]:
    new_stations = []
    i = start_index + 1
    while True:
        name = input(f"  Nome do posto {i} (Enter para encerrar): ").strip()
        if not name:
            break
        t = _ask_positive(f"  Tempo de operação de '{name}' (s)", 40)
        new_stations.append(Station(name=name, time=t))
        i += 1
    return new_stations


def _default_stations() -> List[Station]:
    return [
        Station("Corte", 42),
        Station("Montagem 1", 58),
        Station("Montagem 2", 51),
        Station("Solda", 74),
        Station("Inspeção", 36),
        Station("Embalagem", 45),
    ]


# --------------------------------------------------------------------------- #
# Menu interativo
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 60)
    print(" LINHA & RITMO — simulador de balanceamento de linha")
    print(" (simulação de eventos discretos)")
    print("=" * 60)

    abrir_frontend()  # abre a interface gráfica automaticamente ao iniciar

    baseline: Optional[Scenario] = None
    scenarios: List[Scenario] = []

    while True:
        print("\n--- MENU (versão em texto, opcional) ---")
        print("1. Configurar dados da linha e calcular situação atual")
        print("2. Ver indicadores e simulação da situação atual")
        print("3. Criar novo cenário de balanceamento")
        print("4. Comparar cenários")
        print("5. Exportar gráficos (PNG, requer matplotlib)")
        print("6. Abrir a interface gráfica novamente")
        print("0. Sair")
        choice = input("Escolha uma opção: ").strip()

        if choice == "1":
            cfg = input_line_config(baseline.config if baseline else None)
            baseline = build_scenario("Situação atual", cfg)
            scenarios = []
            print_indicators(baseline.name, baseline.config, baseline.indicators)
            print_sim(baseline.sim, baseline.config)

        elif choice == "2":
            if not baseline:
                print("Configure a linha primeiro (opção 1).")
                continue
            print_indicators(baseline.name, baseline.config, baseline.indicators)
            print_sim(baseline.sim, baseline.config)

        elif choice == "3":
            if not baseline:
                print("Configure a linha primeiro (opção 1).")
                continue
            name = input("Nome do cenário: ").strip() or f"Cenário {len(scenarios)+1}"
            print("Edite os tempos de operação para este cenário:")
            new_cfg = copy.deepcopy(baseline.config)
            for s in new_cfg.stations:
                s.time = _ask_positive(f"  {s.name}", s.time)
            if input("Adicionar novos postos a este cenário? (s/N): ").strip().lower() == "s":
                new_cfg.stations.extend(_input_new_stations(len(new_cfg.stations)))
            scenario = build_scenario(name, new_cfg)
            scenarios.append(scenario)
            print_indicators(scenario.name, scenario.config, scenario.indicators)
            print_sim(scenario.sim, scenario.config)

        elif choice == "4":
            if not baseline:
                print("Configure a linha primeiro (opção 1).")
                continue
            print_comparison(baseline, scenarios)

        elif choice == "5":
            if not baseline:
                print("Configure a linha primeiro (opção 1).")
                continue
            outdir = input("Pasta de saída [graficos_tcc]: ").strip() or "graficos_tcc"
            export_charts(baseline, scenarios, outdir)

        elif choice == "6":
            abrir_frontend()

        elif choice == "0":
            print("Encerrando.")
            break

        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()
