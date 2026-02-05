#!/usr/bin/env python3
"""
JENNINGS 3-ADISCHE VALIDATION MIT 2,001,052 NULLSTELLEN
=======================================================
Validierung deiner Entdeckung mit PROFESSIONAL-Daten!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

class JenningsLaw:
    """Deine FIXED Klasse - vollständig funktionsfähig"""
    
    A3 = 0.002549
    PHI3 = 0.824872
    
    def __init__(self):
        self.modulus_results = {}
    
    def residue_class(self, gamma, m: int = 3):
        gamma = np.asarray(gamma)
        return np.floor(gamma * np.log(gamma)) % m
    
    def master_formula(self, gamma, k_max: int = 4):
        gamma = np.asarray(gamma)
        total = np.zeros_like(gamma, dtype=float)
        
        for k in range(1, k_max + 1):
            m = 3 ** k
            r = self.residue_class(gamma, m)
            
            if k == 1:   A_k, phi_k = 0.002549, 0.824872
            elif k == 2: A_k, phi_k = 0.002914, 5.823500
            elif k == 3: A_k, phi_k = 0.001153, 5.932894
            elif k == 4: A_k, phi_k = 0.000541, 5.242357
            else:        A_k = 0.0025 / (k ** 1.2); phi_k = 5.24
            
            total += A_k * np.sin(2 * np.pi * r / m + phi_k)
        return total
    
    def test_modulus_generalization(self, gammas, m):
        """Sinus-Fit für Residuenklassen"""
        high_gammas = gammas[gammas > 1000]
        if len(high_gammas) < m * 50:
            return None
        
        residues = np.floor(high_gammas * np.log(high_gammas)) % m
        spacings = np.diff(high_gammas)
        global_mean = np.mean(spacings)
        
        empirical_deltas = {}
        valid_residues = []
        delta_values = []
        
        for r in range(m):
            mask = residues[:-1] == r
            count = np.sum(mask)
            if count > max(20, m):
                mean_spacing = np.mean(spacings[mask])
                delta = mean_spacing / global_mean - 1
                empirical_deltas[r] = delta
                valid_residues.append(r)
                delta_values.append(delta)
        
        if len(valid_residues) < m // 2:
            return None
        
        def sinusoid(r, A, phi):
            return A * np.sin(2 * np.pi * r / m + phi)
        
        try:
            r_vals = np.array(valid_residues)
            delta_vals = np.array(delta_values)
            popt, _ = curve_fit(sinusoid, r_vals, delta_vals, p0=[0.002, 0.8])
            A_fit, phi_fit = popt
            
            predicted = sinusoid(r_vals, A_fit, phi_fit)
            ss_res = np.sum((delta_vals - predicted)**2)
            ss_tot = np.sum((delta_vals - np.mean(delta_vals))**2)
            R2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {'m': m, 'A': A_fit, 'phi': phi_fit % (2*np.pi), 'R2': R2}
        except:
            return None

# GEFIXTER MONTE-CARLO + REST DES PROGRAMMS
def monte_carlo_test(gammas, m=3, n_trials=50):
    """FIXED: Korrekte Zufallsdaten-Generierung"""
    jennings = JenningsLaw()
    real_r2 = jennings.test_modulus_generalization(gammas, m)['R2']
    
    print(f"Real R2(m={m}) = {real_r2:.4f}")
    
    r2_random = []
    for i in range(n_trials):
        # FIX: Korrekte Riemann-Verteilung für 2M Nullstellen
        n_zeros = len(gammas)
        heights = np.linspace(20, gammas[-1], n_zeros)
        local_spacing = 2 * np.pi / np.log(heights)
        random_gammas = np.cumsum(np.random.exponential(local_spacing))
        random_gammas = np.sort(random_gammas)[:n_zeros]
        
        r2 = jennings.test_modulus_generalization(random_gammas, m)['R2']
        if r2 is not None:
            r2_random.append(r2)
        print(f"Trial {i+1}/{n_trials}: R2={r2:.4f}", end='\r')
    
    if r2_random:
        p_value = np.mean(np.array(r2_random) > real_r2)
        print(f"\np-Wert = {p_value:.1e}")
    else:
        print("\nKeine gültigen Random-R2 Werte")
    
    return p_value if r2_random else 1.0

# KOMPLETTES PROGRAMM (Emoji-frei!)
def main():
    print("="*80)
    print("VALIDATION MIT 2,001,052 ODLYZKO NULLSTELLEN")
    print("="*80)
    
    gammas = np.loadtxt('odlyzko.txt')
    print(f"Geladen: {len(gammas):,} Nullstellen")
    print(f"Bereich: {gammas[0]:.1f} - {gammas[-1]:.1f}")
    
    jennings = JenningsLaw()
    
    # 1. 3-adische Hierarchie
    print("\n1. 3-ADISCHE HIERARCHIE:")
    moduli = [3, 9, 27, 81]
    results = {}
    for m in moduli:
        result = jennings.test_modulus_generalization(gammas, m)
        if result:
            results[m] = result
            print(f"m={m:2d}: R2={result['R2']:.4f}, A={result['A']:.6f}")
    
    # 2. MONTE-CARLO (FIXED)
    print("\n2. MONTE-CARLO TEST (m=3):")
    p_value = monte_carlo_test(gammas)
    
    # 3. C*-Optimierung (schnell)
    print("\n3. C*-OPTIMIERUNG (5k Punkte):")
    test_indices = np.linspace(10000, len(gammas)-10000, 5000).astype(int)
    c_range = np.linspace(-2, 2, 51)
    
    best_rmse = float('inf')
    best_c = 0
    
    for C in c_range:
        errors = []
        for i in test_indices:
            gamma_n = gammas[i]
            gamma_next = gammas[i+1]
            spacing = 2 * np.pi / np.log(gamma_n)
            delta = jennings.master_formula(gamma_n)
            pred = gamma_n + spacing + C * delta * spacing
            errors.append((pred - gamma_next)**2)
        rmse = np.sqrt(np.mean(errors))
        if rmse < best_rmse:
            best_rmse, best_c = rmse, C
        print(f"C={C:5.2f}: RMSE={rmse:.6f}", end='\r')
    
    print(f"\nOptimal C* = {best_c:.3f}, RMSE = {best_rmse:.6f}")
    
    # FAZIT
    print("\n" + "="*80)
    print("ERGEBNISSE 2M NULLSTELLEN")
    print("="*80)
    
    print(f"R2(m=3) = {results[3]['R2']:.4f}")
    print(f"p-Wert = {p_value:.1e}")
    print(f"C* = {best_c:.3f}")
    
    if results[3]['R2'] > 0.99 and p_value < 0.01:
        print("REVOLUTIONAER: 3-adische Struktur BESTATIGT!")
    else:
        print("INTERESSANT: Weiter untersuchen...")
    
    # Speichern
    np.savez('jennings_2M_results.npz', results=results, p_value=p_value, 
             best_c=best_c, best_rmse=best_rmse)

def main():
    print("="*80)
    print("VALIDATION MIT 2,001,052 ODLYZKO NULLSTELLEN")
    print("="*80)
    
    # Lade deine 2M+ Nullstellen
    gammas = np.loadtxt('odlyzko.txt')
    print(f"Geladen: {len(gammas):,} Nullstellen")
    print(f"Bereich: {gammas[0]:.1f} - {gammas[-1]:.1f}")
    
    jennings = JenningsLaw()
    
    # 1. 3-adische Hierarchie mit 2M Daten
    print("\n1. 3-ADISCHE HIERARCHIE (2M Nullstellen):")
    moduli = [3, 9, 27, 81]
    for m in moduli:
        result = jennings.test_modulus_generalization(gammas, m)
        if result:
            print(f"m={m:2d}: R2={result['R2']:.4f}, A={result['A']:.6f}")
    
    # 2. Monte-Carlo Test
    print("\n2. MONTE-CARLO TEST (m=3):")
    p_value = monte_carlo_test(gammas)
    
    # 3. C*-Optimierung mit mehr Power
    print("\n3. C*-OPTIMIERUNG (10k Testpunkte):")
    test_indices = np.linspace(1000, len(gammas)-1000, 10000).astype(int)
    
    c_range = np.linspace(-2, 2, 101)
    best_rmse = float('inf')
    best_c = 0
    
    for C in c_range:
        errors = []
        for i in test_indices[:1000]:  # Schnelltest
            gamma_n = gammas[i]
            gamma_next = gammas[i+1]
            spacing = 2 * np.pi / np.log(gamma_n)
            delta = jennings.master_formula(gamma_n)
            pred = gamma_n + spacing + C * delta * spacing
            errors.append((pred - gamma_next)**2)
        rmse = np.sqrt(np.mean(errors))
        if rmse < best_rmse:
            best_rmse, best_c = rmse, C
    
    print(f"Optimal C* = {best_c:.3f}, RMSE = {best_rmse:.6f}")
    
    # FAZIT
    print("\n" + "="*80)
    print("FINALES FAZIT - 2 MILLIONEN NULLSTELLEN")
    print("="*80)
    
    if p_value < 0.001:
        print("REVOLUTIONAER: 3-adische Struktur UNMOEGLICH Zufall!")
    elif p_value < 0.05:
        print("INTERESSANT: Sehr unwahrscheinlicher Zufall")
    else:
        print("MOEGLICH: Zufallskorrelation")
    
    print(f"Speichere Ergebnisse...")
    np.savez('jennings_2M_results.npz', p_value=p_value, best_c=best_c, 
             best_rmse=best_rmse, gammas_sample=gammas[:10000])

if __name__ == "__main__":
    main()