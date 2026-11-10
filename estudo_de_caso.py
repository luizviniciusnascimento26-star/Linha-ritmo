"""Estudo de caso - execução das replicações (Apêndice A).
Requer o arquivo linha_ritmo.py na mesma pasta."""
import math
import statistics as st
from linha_ritmo import Station, LineConfig, calc_indicators, simulate

JORNADA, DEMANDA, VARIAB, NPECAS, REPLICACOES = 8, 400, 12, 400, 30
NOMES = ["P1 – Base e coluna", "P2 – Motor", "P3 – Hélice e grades",
         "P4 – Elétrica", "P5 – Painel e teste", "P6 – Expedição"]
CENARIOS = {"Situação atual":   [48, 76, 40, 84, 60, 52],
            "Cenário proposto": [62, 62, 64, 60, 60, 52]}
T_CRIT = 2.045  # t de Student (0,975; 29 graus de liberdade)


def resumo(valores):
    media = st.mean(valores)
    semi = T_CRIT * st.stdev(valores) / math.sqrt(len(valores))
    return f"{media:.2f} ± {semi:.2f}"


for nome, tempos in CENARIOS.items():
    cfg = LineConfig(JORNADA, DEMANDA, VARIAB, NPECAS,
                     [Station(n, t) for n, t in zip(NOMES, tempos)])
    ind = calc_indicators(cfg)
    sims = [simulate(cfg, seed=s) for s in range(1, REPLICACOES + 1)]
    print(f"\n{nome}")
    print(f"  Takt time: {ind.takt_time:.1f} s | Ciclo: {ind.cycle_time} s")
    print(f"  Eficiência: {ind.eficiencia:.1f}%")
    print(f"  Capacidade: {ind.capacidade} un./dia")
    print(f"  Throughput (un./h): {resumo([r.throughput_hora for r in sims])}")
    print(f"  Lead time (s): {resumo([r.avg_lead_time for r in sims])}")
    print(f"  Tempo total (h): {resumo([r.makespan / 3600 for r in sims])}")
    for j, n in enumerate(NOMES):
        u = resumo([r.utilization[j] for r in sims])
        w = resumo([r.avg_wait[j] for r in sims])
        print(f"  {n}: utilização {u} % | fila {w} s")
