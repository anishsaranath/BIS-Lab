import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.linspace(-5, 5, 60)
Y = X**2 + 2*X + 1 + np.random.normal(0, 1.5, size=len(X))

FUNCS = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b}
TERMS = ['x', '1', '2']
HEAD = 5
TAIL = HEAD + 1
CHROM_LEN = HEAD + TAIL

def random_chrom():
    head = list(np.random.choice(list(FUNCS.keys()) + TERMS, HEAD))
    tail = list(np.random.choice(TERMS, TAIL))
    return head + tail

def eval_chrom(chrom, x):
    idx = 0
    def parse():
        nonlocal idx
        sym = chrom[idx]
        idx += 1
        if sym in FUNCS:
            return FUNCS[sym](parse(), parse())
        return x if sym == 'x' else float(sym)
    return parse()

def expr_str(chrom):
    idx = 0
    def parse():
        nonlocal idx
        sym = chrom[idx]
        idx += 1
        if sym in FUNCS:
            return f"({parse()}{sym}{parse()})"
        return sym
    return parse()

pop = [random_chrom() for _ in range(80)]
best_chrom = None
best_err = float('inf')

print("Target Formula : (x^2 + 2x + 1)")
print("-" * 55)

for gen in range(1, 121):
    errors = []
    for c in pop:
        pred = eval_chrom(c, X)
        err = np.mean((pred - Y)**2)
        errors.append(err)
        if err < best_err:
            best_err = err
            best_chrom = c[:]

    if gen % 30 == 0 or gen == 1:
        print(f"Gen {gen:3d} | MSE: {best_err:6.3f} | Formula: {expr_str(best_chrom)}")

    ranked = [pop[i] for i in np.argsort(errors)]
    next_pop = ranked[:10]
    while len(next_pop) < len(pop):
        p1, p2 = ranked[np.random.randint(25)], ranked[np.random.randint(25)]
        cut = np.random.randint(1, CHROM_LEN)
        child = p1[:cut] + p2[cut:]
        if np.random.rand() < 0.3:
            pos = np.random.randint(CHROM_LEN)
            pool = list(FUNCS.keys()) + TERMS if pos < HEAD else TERMS
            child[pos] = np.random.choice(pool)
        next_pop.append(child)
    pop = next_pop

print("-" * 55)
print(f"Best Chromosome (DNA) : {' '.join(best_chrom)}")
print(f"Discovered Formula    : {expr_str(best_chrom)}")
print(f"Final Mean Sq Error   : {best_err:.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(X, Y, color='#2b5c8f', alpha=0.75, label='Noisy Data')
plt.plot(X, eval_chrom(best_chrom, X), color='#e63946', linewidth=2.5, label=f'GEP Formula: {expr_str(best_chrom)}')
plt.title('Gene Expression Programming: Formula Discovery')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
