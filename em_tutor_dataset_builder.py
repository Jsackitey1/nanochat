import json
import random
import math

# Constants
K = 8.99e9 # N*m^2/C^2
EPSILON_0 = 8.85e-12 # C^2/N*m^2

def format_sci(val, precision=2):
    """Formats a number in scientific notation like 2.0e-6."""
    s = "{:.{}e}".format(val, precision)
    base, exponent = s.split('e')
    return f"{base}e{int(exponent)}"

def validate_example(ex):
    required = ["id", "topic", "level", "prompt", "response"]
    for k in required:
        if k not in ex:
            raise ValueError(f"Missing key: {k}")
        if not isinstance(ex[k], str) or not ex[k].strip():
            raise ValueError(f"Key '{k}' must be a non-empty string")

def generate_coulomb_problem(id_counter):
    """Generates a Coulomb's Law problem (1D)."""
    q1_val = random.randint(1, 10) * (10 ** -6) # 1 to 10 microC
    q2_val = random.randint(1, 10) * (10 ** -6) # 1 to 10 microC
    is_attractive = random.choice([True, False])
    
    if is_attractive:
        q2_sign = -1
    else:
        q2_sign = 1
        
    r_val = random.randint(1, 100) / 100.0 # 0.01 to 1.0 m
    
    force = (K * abs(q1_val) * abs(q2_val)) / (r_val ** 2)
    
    q1_str = format_sci(q1_val)
    q2_str = format_sci(q2_val * q2_sign)
    r_str = f"{r_val:.2f}"
    f_str = format_sci(force)
    
    direction = "attractive" if is_attractive else "repulsive"
    
    prompt = (
        "You are a physics E&M tutor. Solve step-by-step and include a unit check.\n\n"
        f"Problem: Two point charges q1 = {q1_str} C and q2 = {q2_str} C are separated by {r_str} m.\n"
        "Find the magnitude of the force between them and state if it is attractive or repulsive.\n"
        "Use k = 8.99e9 N·m^2/C^2."
    )
    
    response = (
        "We use Coulomb’s law: F = k |q1 q2| / r^2.\n"
        f"Identify values: q1 = {q1_str} C, q2 = {q2_str} C, r = {r_str} m, k = 8.99e9 N·m^2/C^2.\n"
        "Substitute into the equation:\n"
        f"F = (8.99e9) * ({q1_str} * {str(abs(q2_val * q2_sign)) + 'e-6' if 'e' not in str(q2_val) else format_sci(abs(q2_val))} ) / ({r_str}^2)\n" # simplified formatting
        f"F = (8.99e9) * ({format_sci(abs(q1_val * q2_val))}) / {r_val**2:.4f}\n" 
        f"F = {f_str} N.\n"
        "Unit check: (N·m^2/C^2) * (C * C) / m^2 = N·m^2/C^2 * C^2 / m^2 = N. The unit is Newtons, which is correct for force.\n"
        f"Since q1 is positive and q2 is {'negative' if is_attractive else 'positive'}, the charges have {'opposite' if is_attractive else 'like'} signs.\n"
        f"Answer: {f_str} N ({direction})."
    )
    
    return {
        "id": f"em_{id_counter:04d}",
        "topic": "Coulomb",
        "level": "intro",
        "prompt": prompt,
        "response": response
    }

def generate_efield_point_problem(id_counter):
    q_val = random.randint(1, 20) * (10 ** -9) # nC
    r_val = random.choice([0.1, 0.2, 0.5, 1.0, 2.0])
    
    e_field = (K * q_val) / (r_val ** 2)
    q_str = format_sci(q_val)
    e_str = format_sci(e_field)
    
    prompt = (
        "You are a physics E&M tutor. Solve step-by-step and include a unit check.\n\n"
        f"Problem: Calculate the magnitude of the electric field at a distance r = {r_val} m from a point charge q = {q_str} C.\n"
        "Use k = 8.99e9 N·m^2/C^2."
    )
    
    response = (
        "The formula for the electric field of a point charge is E = k |q| / r^2.\n"
        f"Given: k = 8.99e9 N·m^2/C^2, q = {q_str} C, r = {r_val} m.\n"
        f"E = (8.99e9 * {q_str}) / ({r_val}^2)\n"
        f"E = {format_sci(K*q_val)} / {r_val**2}\n"
        f"E = {e_str} N/C.\n"
        "Unit check: (N·m^2/C^2) * C / m^2 = N·m^2/C / m^2 = N/C. Correct.\n"
        f"Answer: {e_str} N/C."
    )
    
    return {
        "id": f"em_{id_counter:04d}",
        "topic": "Electric Field",
        "level": "intro",
        "prompt": prompt,
        "response": response
    }

def generate_capacitor_energy(id_counter):
    c_val = random.randint(1, 100) * (10 ** -6) # uF
    v_val = random.randint(5, 50) # V
    
    energy = 0.5 * c_val * (v_val ** 2)
    
    c_str = format_sci(c_val)
    e_str = format_sci(energy)
    
    prompt = (
        "You are a physics E&M tutor. Solve step-by-step and include a unit check.\n\n"
        f"Problem: A capacitor with capacitance C = {c_str} F is charged to a potential difference of V = {v_val} V. Calculate the energy stored in the capacitor."
    )
    
    response = (
        "The energy stored in a capacitor is given by U = (1/2)CV^2.\n"
        f"Given: C = {c_str} F, V = {v_val} V.\n"
        f"U = 0.5 * {c_str} * ({v_val}^2)\n"
        f"U = 0.5 * {c_str} * {v_val**2}\n"
        f"U = {e_str} J.\n"
        "Unit check: F * V^2 = (C/V) * V^2 = C * V = J. Correct.\n"
        f"Answer: {e_str} J."
    )
    
    return {
        "id": f"em_{id_counter:04d}",
        "topic": "Capacitance",
        "level": "intro",
        "prompt": prompt,
        "response": response
    }

def generate_gauss_sphere(id_counter):
    q_val = random.randint(1, 10) * (10 ** -6) # uC
    r_val = random.randint(10, 50) / 100.0 # m
    
    flux = q_val / EPSILON_0 
    
    q_str = format_sci(q_val)
    flux_str = format_sci(flux)
    
    prompt = (
        "You are a physics E&M tutor. Solve step-by-step and include a unit check.\n\n"
        f"Problem: A point charge q = {q_str} C is located at the center of a spherical surface of radius r = {r_val} m. "
        "What is the net electric flux through the sphere?\n"
        "Use epsilon_0 = 8.85e-12 C^2/(N·m^2)."
    )
    
    response = (
        "According to Gauss's Law, the net electric flux through a closed surface depends only on the enclosed charge: Phi = q_enclosed / epsilon_0.\n"
        "The radius of the sphere does not affect the total flux, provided the charge is inside.\n"
        f"Given: q_enclosed = {q_str} C, epsilon_0 = 8.85e-12 C^2/(N·m^2).\n"
        f"Phi = {q_str} / 8.85e-12\n"
        f"Phi = {flux_str} N·m^2/C.\n"
        "Unit check: C / (C^2/(N·m^2)) = C * (N·m^2/C^2) = N·m^2/C. Correct.\n"
        f"Answer: {flux_str} N·m^2/C."
    )
    
    return {
        "id": f"em_{id_counter:04d}",
        "topic": "Gauss's Law",
        "level": "intro",
        "prompt": prompt,
        "response": response
    }

def main():
    examples = []
    
    # Generate 250 examples
    count = 0
    generators = [generate_coulomb_problem, generate_efield_point_problem, generate_capacitor_energy, generate_gauss_sphere]
    
    for i in range(250):
        count += 1
        gen = random.choice(generators)
        examples.append(gen(count))
        
    # Split into train (200) and test (50)
    train_ex = examples[:200]
    test_ex = examples[200:]
    
    with open("em_tutor_train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
    with open("em_tutor_test.jsonl", "w", encoding="utf-8") as f:
        for ex in test_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(train_ex)} training examples and {len(test_ex)} test examples.")

if __name__ == "__main__":
    main()
