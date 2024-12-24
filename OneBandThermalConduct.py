# sum over n=1~N
import numpy as np

sum_n = 0
                delta = eigval / (0.0862*T)
                rho_null = 1/(np.exp(delta)-1)
                i = 0
                while i < 4:

                    if eigval[i] > 0:
                        data = C2.subs([(rho, rho_null)]) * np.real(mat_omega_nk[i])
                        sum_n = sum_n + data
                    else:
                        pass
                    i += 1