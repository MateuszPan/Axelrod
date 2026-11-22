# Replikacja modelu dyfuzji kulturowej Axelroda

Jest to replikacja w Pythonie dwóch wyników z artykułu Roberta Axelroda (1997) *The Dissemination of Culture: A Model with Local Convergence and Global Polarization*: **tabeli 2** i **rysunku 2**. Założeniem nie było uzyskanie dokładnie takich samych wyników jak Axelrod. Chcieliśmy przede wszystkim samodzielnie odtworzyć zasady działania jego modelu i sprawdzić, czy nasze symulacje prowadzą do podobnych wniosków.

Model opisuje agentów na siatce `n × n`. Każdy agent ma `F` cech kulturowych, a każda z nich przyjmuje jedną z `q` wartości. Sąsiedzi wchodzą w interakcje z prawdopodobieństwem, które zależy od podobieństwa kulturowego. Symulacja kończy się po osiągnięciu stabilności.

## Wymagania i instalacja
- Python 3.11 (projekt uruchamiany z Pythonem 3.11.9)
- Biblioteki, które są wymienione w `requirements.txt`, w tym NumPy, pandas, Matplotlib i Numba

W głównym folderze projektu, w terminalu PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Uruchamianie

```powershell
.\.venv\Scripts\python.exe -m tests.test_model
.\.venv\Scripts\python.exe -m experiments.table2
.\.venv\Scripts\python.exe -m experiments.table2_summary
.\.venv\Scripts\python.exe -m experiments.table2_dynamics_summary
.\.venv\Scripts\python.exe -m experiments.table2_plots
.\.venv\Scripts\python.exe -m experiments.figure2
.\.venv\Scripts\python.exe -m experiments.figure2_summary
.\.venv\Scripts\python.exe -m experiments.figure2_plots
```

**Uwaga:** Jeśli plik `experiments.table2` zostanie ponownie uruchomiony, to nadpisuje on plik `data/table2/table2_runs.csv`. Dlatego przed uruchomieniem warto zrobić kopię wyników. `experiments.figure2` pomija ukończone przebiegi i dopisuje brakujące; duże siatki mogą wymagać długich obliczeń.

## Wyniki

| Eksperyment | Parametry | Liczba przebiegów | Najważniejszy wynik |
|---|---|---:|---|
| Tabela 2 | `n=10`; `F,q ∈ {5,10,15}` | 90 | Dla `F=5, q=15`: średnio **21,7** regionu (Axelrod: 20,0). |
| Rysunek 2 | `F=5, q=15`; wybrane `n` od 2 do 100 | 380 | Maksimum **21,3** regionu przy `n=12`; spadek do **3,2** przy `n=100`. |

Dane znajdują się w `data/table2/` i `data/figure2/`, a wykresy w `plots/table2/` i `plots/figure2/`. Kod modelu znajduje się w `model.py`, a testy w `tests/`. Stałe ziarna losowania umożliwiają odtwarzanie wyników. Wartości z oryginalnego rysunku 2 odczytano w przybliżeniu.

## Źródło

Axelrod, R. (1997). *The Dissemination of Culture: A Model with Local Convergence and Global Polarization*. Journal of Conflict Resolution, 41 (2), 203–226. https://doi.org/10.1177/0022002797041002001

Autorzy:
Mateusz Pandura i Marcin Kochalski

English version: [README.md](README.md).
