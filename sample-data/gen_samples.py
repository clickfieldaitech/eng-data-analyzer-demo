import csv, math, random

def gauss(mean, std):
    return random.gauss(mean, std)

def round_(v, d):
    return round(v, d)

# ---- Hydrostatic pressure test ----
random.seed(1337)
rows = []
for t in range(0, 101):
    if t <= 30:
        p = 25*(t/30) - 0.02*((30-t)**1.2)/10
    elif t <= 90:
        p = 25
    else:
        p = 25*(1-(t-90)/10)
    p = max(0, p + gauss(0, 0.06))
    disp = max(0, (p**1.04)*0.0168 + gauss(0, 0.004))
    stress = max(0, p*5.05 + gauss(0, 0.8))
    if 30 < t < 90:
        temp = 24.8 + (t-30)/60*0.9 + gauss(0, 0.05)
    elif t >= 90:
        temp = 24.8 + 0.9 - ((t-90)/10*0.4) + gauss(0, 0.05)
    else:
        temp = 24.8 + gauss(0, 0.05)
    if t == 62:
        p += 0.4
    rows.append([round_(t,1), round_(p,2), round_(disp,3), round_(stress,1), round_(temp,2)])

with open('/Users/kishorekv/eng-data-analyzer-demo/sample-data/Bellows_Hydro_Test_04.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Time (s)', 'Pressure (bar)', 'Axial Displacement (mm)', 'Hoop Stress (MPa)', 'Temperature (C)'])
    w.writerows(rows)

# ---- Cyclic fatigue test ----
random.seed(4242)
rows = []
for c in range(0, 10001, 50):
    frac = c/10000
    base = 142.5 - 14*(frac**1.6) - (22*(((frac-0.8)/0.2)**2) if frac > 0.8 else 0)
    amp = max(90, base + gauss(0, 1.1))
    disp = 0.412 - 0.03*(frac**1.6) + gauss(0, 0.003)
    pmin = 2.05 + gauss(0, 0.04)
    pmax = 24.8 - 0.3*(frac**2) + gauss(0, 0.05)
    status = 'OK' if amp > 135 else ('Degrading' if amp > 125 else 'Near limit')
    rows.append([c, round_(pmin,2), round_(pmax,2), round_(amp,1), round_(disp,3), status])

with open('/Users/kishorekv/eng-data-analyzer-demo/sample-data/Bellows_Fatigue_Test_11.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Cycle', 'Pressure Min (bar)', 'Pressure Max (bar)', 'Stress Amplitude (MPa)', 'Displacement (mm)', 'Status'])
    w.writerows(rows)

# ---- Spring rate test ----
random.seed(777)
rows = []
k = 81.4
for i in range(0, 33):
    d = round_(i*0.25, 2)
    loading = k*d + 3.1*(d**1.3) + gauss(0, 1.4)
    hyst = 26*math.sin(math.pi*(d/8))
    unloading = max(0, loading - hyst + gauss(0, 1.2))
    if i == 0:
        loading = 0
        unloading = 0
    rows.append([d, round_(loading,1), round_(unloading,1)])

with open('/Users/kishorekv/eng-data-analyzer-demo/sample-data/Bellows_SpringRate_B2026-11.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Displacement (mm)', 'Force Loading (N)', 'Force Unloading (N)'])
    w.writerows(rows)

print("done")
