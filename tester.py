import math

def simulate(m1, m2, v0=1.0, max_steps=10000000):
    v1 = 0.0
    v2 = -abs(v0)
    count = 0
    for _ in range(max_steps):
        if v1 < 0:
            v1 = -v1
            count += 1
            continue
        if v2 < v1:
            v1_new = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
            v2_new = (2 * m1 * v1 + (m2 - m1) * v2) / (m1 + m2)
            v1, v2 = v1_new, v2_new
            count += 1
            continue
        break
    return count, v1, v2

if __name__ == "__main__":
    print("n\tratio\tN\tfloor(pi*sqrt(r))\tmatch")
    for n in range(1, 11):
        m1 = 1.0
        m2 = 10.0 ** n
        N, v1f, v2f = simulate(m1, m2, 1.0)
        formula = math.floor(math.pi * math.sqrt(m2 / m1))
        print(f"{n}\t{int(m2)}\t{N}\t{formula}\t{'YES' if N == formula else 'NO'}")
