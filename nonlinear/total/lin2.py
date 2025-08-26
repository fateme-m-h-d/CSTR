import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, log, exp

# === Original Parameters ===
Cao, Cbo, Cco = 1, 2, 0  # mol/L
V, Q = 10, 1  # L, L/s
tau = V/Q  # s

# Kinetic parameters
Afo = 1e13  # Note: fixed from 10e12 to 1e13
Eaf = 90000  # J/mol
Aro = 1e11  # Note: fixed from 10e10 to 1e11
Ear = 80000  # J/mol
R = 8.314  # J/mol·K

# Original steady state
Cass_orig = 0.3510168217392415
Cbss_orig = 0.7020336434749113
Tss_orig = 550.6212424849699

print("=== LOGARITHMIC TRANSFORMATION APPROACHES ===\n")

def approach1_inverse_temperature():
    """Approach 1: Transform T → u = 1/T"""
    print("1. INVERSE TEMPERATURE TRANSFORMATION: u = 1/T")
    print("   This makes exp(-E/RT) = exp(-E*R*u) which is linear in u")
    
    # Define new variables
    Ca, Cb, u = symbols('Ca Cb u')  # u = 1/T
    
    # Express temperature in terms of u
    # T = 1/u, but we need to be careful about units and numerical stability
    # Let's use u = 1000/T to keep u values reasonable
    T_from_u = 1000/u  # T in Kelvin when u is in 1000/K
    
    # Reaction rates in transformed coordinates
    kf = Afo * exp(-Eaf/(R*T_from_u))  # = Afo * exp(-Eaf*u/(R*1000))
    kr = Aro * exp(-Ear/(R*T_from_u))  # = Aro * exp(-Ear*u/(R*1000))
    
    # Material balance in transformed coordinates
    f_u = Cao - Ca - kf*Ca*Cb**2*tau + kr*(Cao-Ca+Cbo-Cb+Cco)*tau
    
    print(f"   u = 1000/T")
    print(f"   kf = Afo * exp(-{Eaf/(R*1000):.6f} * u)")
    print(f"   kr = Aro * exp(-{Ear/(R*1000):.6f} * u)")
    
    # Transform steady state
    uss_orig = 1000/Tss_orig
    print(f"   Original: T_ss = {Tss_orig:.2f} K → u_ss = {uss_orig:.6f}")
    
    # Calculate derivatives in u-space
    df_u_Ca = f_u.diff(Ca)
    df_u_Cb = f_u.diff(Cb) 
    df_u_u = f_u.diff(u)
    
    # Evaluate at steady state
    fss_u = f_u.subs([(Ca, Cass_orig), (Cb, Cbss_orig), (u, uss_orig)])
    df_u_Cass = df_u_Ca.subs([(Ca, Cass_orig), (Cb, Cbss_orig), (u, uss_orig)])
    df_u_Cbss = df_u_Cb.subs([(Ca, Cass_orig), (Cb, Cbss_orig), (u, uss_orig)])
    df_u_uss = df_u_u.subs([(Ca, Cass_orig), (Cb, Cbss_orig), (u, uss_orig)])
    
    # Linearized function in u-space
    f_u_linearized = fss_u + df_u_Cass*(Ca-Cass_orig) + df_u_Cbss*(Cb-Cbss_orig) + df_u_uss*(u-uss_orig)
    
    print(f"   ∂f/∂u at steady state = {float(df_u_uss):.2e}")
    print(f"   Compare to ∂f/∂T = {float(df_u_uss * (-1000/Tss_orig**2)):.2e}")  # Chain rule: ∂f/∂T = ∂f/∂u * ∂u/∂T
    
    print(f"   Linearized: f_u = {f_u_linearized}")
    
    return f_u, f_u_linearized, uss_orig, df_u_uss, df_u_Cass, df_u_Cbss

def approach2_log_rates():
    """Approach 2: Transform the reaction rates logarithmically"""
    print("\n2. LOG REACTION RATES TRANSFORMATION")
    print("   Define w1 = ln(kf), w2 = ln(kr) and linearize in (T, w1, w2) space")
    
    Ca, Cb, T, w1, w2 = symbols('Ca Cb T w1 w2')
    
    # w1 = ln(kf) = ln(Afo) - Eaf/(RT)
    # w2 = ln(kr) = ln(Aro) - Ear/(RT)
    # So: kf = exp(w1), kr = exp(w2)
    
    # The constraint becomes linear in w1, w2:
    # ln(kf) = ln(Afo) - Eaf/(RT)  →  w1 = ln(Afo) - Eaf/(RT)
    # ln(kr) = ln(Aro) - Ear/(RT)  →  w2 = ln(Aro) - Ear/(RT)
    
    # Material balance with rate variables
    f_rates = Cao - Ca - exp(w1)*Ca*Cb**2*tau + exp(w2)*(Cao-Ca+Cbo-Cb+Cco)*tau
    
    # Additional constraints linking rates to temperature
    g1 = w1 - (sym.log(Afo) - Eaf/(R*T))  # g1 = 0
    g2 = w2 - (sym.log(Aro) - Ear/(R*T))  # g2 = 0
    
    # Steady state values
    w1ss = float(sym.log(Afo) - Eaf/(R*Tss_orig))
    w2ss = float(sym.log(Aro) - Ear/(R*Tss_orig))
    
    print(f"   At steady state: w1_ss = ln(kf) = {w1ss:.2f}")
    print(f"   At steady state: w2_ss = ln(kr) = {w2ss:.2f}")
    
    # This gives us 3 constraints instead of 1, but they're much more linear!
    print(f"   Constraint 1: f = {f_rates} = 0")
    print(f"   Constraint 2: g1 = w1 - ln(Afo) + Eaf/(RT) = 0")  
    print(f"   Constraint 3: g2 = w2 - ln(Aro) + Ear/(RT) = 0")
    
    return f_rates, g1, g2, w1ss, w2ss

def approach3_log_transform_direct():
    """Approach 3: Apply log transformation directly to the constraint"""
    print("\n3. DIRECT LOGARITHMIC TRANSFORMATION")
    print("   Transform the constraint f(T,Ca,Cb) = 0 to ln|f| or sign(f)*ln(1+|f|)")
    
    Ca, Cb, T = symbols('Ca Cb T')
    kf = Afo * exp(-Eaf/(R*T))
    kr = Aro * exp(-Ear/(R*T))
    
    # Original constraint
    f_orig = Cao - Ca - kf*Ca*Cb**2*tau + kr*(Cao-Ca+Cbo-Cb+Cco)*tau
    
    # Log-transformed constraint (using signed log for robustness)
    # h = sign(f) * ln(1 + |f|/scale) where scale prevents issues when f ≈ 0
    scale = 1e-6
    f_abs = sym.Abs(f_orig)
    h = sym.sign(f_orig) * sym.log(1 + f_abs/scale)
    
    print(f"   h = sign(f) * ln(1 + |f|/{scale})")
    print(f"   This transforms exponential nonlinearity to logarithmic nonlinearity")
    
    # Calculate derivatives of h
    dh_Ca = h.diff(Ca)
    dh_Cb = h.diff(Cb)
    dh_T = h.diff(T)
    
    print(f"   Note: This approach has mathematical complications due to absolute value")
    print(f"   Better to use approaches 1 or 2")
    
    return h

def approach4_arrhenius_linearization():
    """Approach 4: Linearize Arrhenius terms separately"""
    print("\n4. ARRHENIUS LINEARIZATION")
    print("   Linearize ln(kf) and ln(kr) with respect to 1/T, then exponentiate")
    
    Ca, Cb, T = symbols('Ca Cb T')
    
    # Take logarithm of reaction rates
    ln_kf = sym.log(Afo) - Eaf/(R*T)
    ln_kr = sym.log(Aro) - Ear/(R*T)
    
    # These are linear in 1/T!
    inv_T = 1/T
    print(f"   ln(kf) = ln(Afo) - (Eaf/R) * (1/T) = {sym.log(Afo):.2f} - {Eaf/R:.1f} * (1/T)")
    print(f"   ln(kr) = ln(Aro) - (Ear/R) * (1/T) = {sym.log(Aro):.2f} - {Ear/R:.1f} * (1/T)")
    
    # Linearize ln(kf) and ln(kr) around steady state
    inv_Tss = 1/Tss_orig
    ln_kf_ss = float(sym.log(Afo) - Eaf/(R*Tss_orig))
    ln_kr_ss = float(sym.log(Aro) - Ear/(R*Tss_orig))
    
    # Derivatives of ln(k) with respect to 1/T
    d_ln_kf_d_invT = -Eaf/R
    d_ln_kr_d_invT = -Ear/R
    
    # Linearized log rates
    ln_kf_lin = ln_kf_ss + d_ln_kf_d_invT * (inv_T - inv_Tss)
    ln_kr_lin = ln_kr_ss + d_ln_kr_d_invT * (inv_T - inv_Tss)
    
    # Exponentiate to get linearized rates
    kf_lin = exp(ln_kf_lin)
    kr_lin = exp(ln_kr_lin)
    
    print(f"   At T_ss = {Tss_orig:.1f}K: ln(kf_ss) = {ln_kf_ss:.2f}, ln(kr_ss) = {ln_kr_ss:.2f}")
    
    # Material balance with linearized rates
    f_arrhenius_lin = Cao - Ca - kf_lin*Ca*Cb**2*tau + kr_lin*(Cao-Ca+Cbo-Cb+Cco)*tau
    
    print(f"   f_linearized_arrhenius = {f_arrhenius_lin}")
    
    return f_arrhenius_lin, kf_lin, kr_lin

def compare_approaches():
    """Compare linearization quality of different approaches"""
    print("\n=== COMPARING APPROACHES ===\n")
    
    # Temperature range for testing
    T_test = np.linspace(500, 600, 100)
    
    # Original approach
    Ca, Cb, T = symbols('Ca Cb T') 
    kf = Afo * exp(-Eaf/(R*T))
    kr = Aro * exp(-Ear/(R*T))
    f_orig = Cao - Ca - kf*Ca*Cb**2*tau + kr*(Cao-Ca+Cbo-Cb+Cco)*tau
    
    # Original linearization
    df_T_orig = f_orig.diff(T)
    fss_orig = f_orig.subs([(Ca, Cass_orig), (Cb, Cbss_orig), (T, Tss_orig)])
    df_Tss_orig = df_T_orig.subs([(Ca, Cass_orig), (Cb, Cbss_orig), (T, Tss_orig)])
    f_lin_orig = fss_orig + df_Tss_orig*(T - Tss_orig)
    
    # Inverse temperature approach
    u = symbols('u')
    f_u, f_u_lin, uss_orig, df_u_uss, df_u_Cass, df_u_Cbss = approach1_inverse_temperature()
    
    # Convert back to T coordinates for comparison
    # u = 1000/T, so when comparing we substitute T values
    T_vals = T_test
    u_vals = 1000/T_vals
    
    # Evaluate functions numerically
    f_orig_func = sym.lambdify((Ca, Cb, T), f_orig, 'numpy')
    f_lin_orig_func = sym.lambdify((Ca, Cb, T), f_lin_orig, 'numpy')
    f_u_func = sym.lambdify((Ca, Cb, u), f_u, 'numpy')
    f_u_lin_func = sym.lambdify((Ca, Cb, u), f_u_lin, 'numpy')
    
    # Evaluate at steady state concentrations
    f_orig_vals = f_orig_func(Cass_orig, Cbss_orig, T_vals)
    f_lin_orig_vals = f_lin_orig_func(Cass_orig, Cbss_orig, T_vals)
    f_u_vals = f_u_func(Cass_orig, Cbss_orig, u_vals)
    f_u_lin_vals = f_u_lin_func(Cass_orig, Cbss_orig, u_vals)
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(T_vals, f_orig_vals, 'b-', label='Nonlinear', linewidth=2)
    plt.plot(T_vals, f_lin_orig_vals, 'r--', label='Original linearization', linewidth=2)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(Tss_orig, color='green', linestyle=':', alpha=0.7, label='Steady state')
    plt.xlabel('Temperature (K)')
    plt.ylabel('f(Ca,Cb,T)')
    plt.title('Original Linearization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(T_vals, f_u_vals, 'b-', label='Nonlinear', linewidth=2)
    plt.plot(T_vals, f_u_lin_vals, 'orange', linestyle='--', label='u = 1/T linearization', linewidth=2)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(Tss_orig, color='green', linestyle=':', alpha=0.7, label='Steady state')
    plt.xlabel('Temperature (K)')
    plt.ylabel('f(Ca,Cb,T)')
    plt.title('Inverse Temperature Linearization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linearization_approaches_comparison.png', dpi=300)
    plt.show()
    
    # Calculate errors
    error_orig = np.abs(f_lin_orig_vals - f_orig_vals)
    error_u = np.abs(f_u_lin_vals - f_u_vals)
    
    print(f"Original linearization - Max error: {error_orig.max():.2e}, Mean error: {error_orig.mean():.2e}")
    print(f"u=1/T linearization - Max error: {error_u.max():.2e}, Mean error: {error_u.mean():.2e}")
    
    if error_u.max() < error_orig.max():
        print("✓ Inverse temperature transformation reduces linearization error!")
    else:
        print("⚠️  Inverse temperature transformation doesn't improve linearization")

def practical_implementation():
    """Show how to practically implement this in your neural network"""
    print("\n=== PRACTICAL IMPLEMENTATION FOR NEURAL NETWORK ===\n")
    
    print("To implement u = 1000/T transformation in your neural network:")
    print()
    print("1. MODIFY YOUR DATA PREPROCESSING:")
    print("   # Instead of using T directly")
    print("   T_data = df['Temperature (T)'].values")
    print("   # Transform to u = 1000/T")
    print("   u_data = 1000 / T_data")
    print("   # Use u_data as input to neural network")
    print()
    
    print("2. UPDATE YOUR CONSTRAINT COEFFICIENTS:")
    print("   # Calculate coefficients in u-space using the transformed derivatives")
    u = symbols('u')
    Ca, Cb = symbols('Ca Cb')
    T_from_u = 1000/u
    kf = Afo * exp(-Eaf/(R*T_from_u))
    kr = Aro * exp(-Ear/(R*T_from_u))
    f_u = Cao - Ca - kf*Ca*Cb**2*tau + kr*(Cao-Ca+Cbo-Cb+Cco)*tau
    
    # Calculate derivatives
    df_u_Ca = f_u.diff(Ca)
    df_u_Cb = f_u.diff(Cb)
    df_u_u = f_u.diff(u)
    
    # At steady state
    uss = 1000/Tss_orig
    df_u_Cass = float(df_u_Ca.subs([(Ca, Cass_orig), (Cb, Cbss_orig), (u, uss)]))
    df_u_Cbss = float(df_u_Cb.subs([(Ca, Cass_orig), (Cb, Cbss_orig), (u, uss)]))
    df_u_uss = float(df_u_u.subs([(Ca, Cass_orig), (Cb, Cbss_orig), (u, uss)]))
    fss_u = float(f_u.subs([(Ca, Cass_orig), (Cb, Cbss_orig), (u, uss)]))
    
    # Constraint: A*x + B*z - b = 0 where x = [u], z = [Ca, Cb, Cc]
    A_u = df_u_uss
    B_u = [df_u_Cass, df_u_Cbss, 0.0]
    b_u = df_u_uss*uss + df_u_Cass*Cass_orig + df_u_Cbss*Cbss_orig - fss_u
    
    print(f"   A_list = [torch.tensor([[{A_u:.6f}]])]")
    print(f"   B_list = [torch.tensor([[{B_u[0]:.6f}, {B_u[1]:.6f}, {B_u[2]:.6f}]])]")
    print(f"   b_list = [torch.tensor([{b_u:.6f}])]")
    print()
    
    print("3. MODIFY YOUR NEURAL NETWORK INPUT:")
    print("   # Input is now u instead of T")
    print("   # Network: u → [Ca, Cb, Cc]")
    print("   # When making predictions at temperature T_new:")
    print("   u_new = 1000 / T_new")
    print("   predictions = model(u_new)")
    print()
    
    print("4. FOR MULTIPLE TEMPERATURE RANGES:")
    print("   # Apply the same u = 1000/T transformation to all ranges")
    print("   # Recalculate steady states in u-coordinates")
    print("   # Recalculate A, B, b for each range using transformed derivatives")

if __name__ == "__main__":
    # Run all approaches
    approach1_inverse_temperature()
    approach2_log_rates() 
    approach3_log_transform_direct()
    approach4_arrhenius_linearization()
    
    # Compare approaches
    compare_approaches()
    
    # Show practical implementation
    practical_implementation()
    
    print("\n=== RECOMMENDATION ===")
    print("Use Approach 1 (u = 1000/T transformation):")
    print("✓ Mathematically sound")
    print("✓ Easy to implement")
    print("✓ Converts exponential nonlinearity to much more manageable form")
    print("✓ Single transformation for all temperature ranges")
    print("✓ Maintains physical interpretability")