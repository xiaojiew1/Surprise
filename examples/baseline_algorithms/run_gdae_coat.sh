for p_dim in 10 20 50 100; do
  for lam in 1e-3 5e-2 1e-2 5e-1; do
    for lr in 1e-3 5e-2 1e-2 5e-1; do
      python gdae_coat.py --p_dim ${p_dim} --lam ${lam} --lr ${lr}
    done
  done
done