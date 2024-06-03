import sympy as sp
import numpy as np
from matplotlib.pyplot import subplots as spp
import matplotlib.cm as cm
from numpy import linalg as LA

def fm_spin_model(strg):
    a_i = sp.symbols("a1:3")  # Bravais lattice vectors
    a = sp.Symbol("a", real=True)
    I = sp.I
    pi = sp.pi
    hb = sp.Symbol("ℏ", real=True)
    S = sp.Symbol("S", real=True)

    J = sp.Symbol("J", real=True)  # Heisenberg coupling
    K = sp.Symbol("K", real=True)  # Kitaev coupling
    Gamma = sp.Symbol("Γ", real=True)
    gamma = sp.Symbol("Γ'", real=True)
    h = sp.Symbol("h", real=True)

    V_i = sp.symbols("V1:4")
    V1 = sp.Matrix([1, 0, 0])
    V2 = sp.Matrix([0, 1, 0])
    V3 = sp.Matrix([0, 0, 1])

    Hberg_uu = V1 * sp.Transpose(V1) + V2 * sp.Transpose(V2) + V3 * sp.Transpose(V3)

    IsingXX_uu = V1 * sp.Transpose(V1)
    IsingYY_uu = V2 * sp.Transpose(V2)
    IsingZZ_uu = V3 * sp.Transpose(V3)

    XY_uu = V1 * sp.Transpose(V2)
    XZ_uu = V1 * sp.Transpose(V3)
    YX_uu = V2 * sp.Transpose(V1)
    YZ_uu = V2 * sp.Transpose(V3)
    ZX_uu = V3 * sp.Transpose(V1)
    ZY_uu = V3 * sp.Transpose(V2)

    Hx_uu = 2 * K * IsingXX_uu + J * Hberg_uu + Gamma * (YZ_uu + ZY_uu) + gamma * (YX_uu + ZX_uu + XY_uu + XZ_uu)
    Hy_uu = 2 * K * IsingYY_uu + J * Hberg_uu + Gamma * (XZ_uu + ZX_uu) + gamma * (YZ_uu + YX_uu + ZY_uu + XY_uu)
    Hz_uu = 2 * K * IsingZZ_uu + J * Hberg_uu + Gamma * (XY_uu + YX_uu) + gamma * (XZ_uu + YZ_uu + ZX_uu + ZY_uu)

    e = sp.Symbol("e")
    f = sp.Symbol("f")
    g = sp.Symbol("g")
    h = sp.Symbol("h")
    n_i = sp.Symbol("𝑛𝑖")
    n_j = sp.Symbol("𝑛𝑗")

    HPXX = 1 / sp.Integer(2) * (e + f + g + h)
    HPYY = -1 / sp.Integer(2) * (e - f - g + h)
    HPZZ = (S - n_i - n_j)
    HPXY = 1 / (2 * I) * (e - f + g - h)
    HPYX = 1 / (2 * I) * (e + f - g - h)

    # for now we didn't how to calculate single a, a+...
    HPXZ = 0
    HPYZ = 0
    HPZX = 0
    HPZY = 0

    HPbosons = sp.Matrix([[HPXX, HPXY, HPXZ], [HPYX, HPYY, HPYZ], [HPZX, HPZY,
                                                                   HPZZ]])
    o = sp.Integer(0)
    ii = sp.Integer(1)

    HPHx_uu = sp.Integer(0)

    i = 0
    while i < 3:
        j = 0
        while j < 3:
            HPHx_uu = HPHx_uu + Hx_uu[i, j] * HPbosons[i, j]
            j += 1
        i += 1

    HPHx_uu = HPHx_uu.simplify()

    Coef_HPHx_uu = sp.Matrix([o, o, o, o, o, o])
    Coef_HPHx_uu[0, 0] = HPHx_uu.coeff(e)
    Coef_HPHx_uu[1, 0] = HPHx_uu.coeff(f)
    Coef_HPHx_uu[2, 0] = HPHx_uu.coeff(g)
    Coef_HPHx_uu[3, 0] = HPHx_uu.coeff(h)
    Coef_HPHx_uu[4, 0] = HPHx_uu.coeff(n_i)
    Coef_HPHx_uu[5, 0] = HPHx_uu.coeff(n_j)

    Mat_Coef_HPHx_uu = sp.Matrix([
        [Coef_HPHx_uu[4, 0], Coef_HPHx_uu[1, 0], o, Coef_HPHx_uu[3, 0]],
        [Coef_HPHx_uu[1, 0], Coef_HPHx_uu[5, 0], Coef_HPHx_uu[3, 0], o],
        [o, Coef_HPHx_uu[0, 0], Coef_HPHx_uu[5, 0], Coef_HPHx_uu[2, 0]],
        [Coef_HPHx_uu[0, 0], o, Coef_HPHx_uu[2, 0], Coef_HPHx_uu[4, 0]]
    ])

    HPHy_uu = sp.Integer(0)

    i = 0
    while i < 3:
        j = 0
        while j < 3:
            HPHy_uu = HPHy_uu + Hy_uu[i, j] * HPbosons[i, j]
            j += 1
        i += 1

    HPHy_uu = HPHy_uu.simplify()

    Coef_HPHy_uu = sp.Matrix([o, o, o, o, o, o])
    Coef_HPHy_uu[0, 0] = HPHy_uu.coeff(e)
    Coef_HPHy_uu[1, 0] = HPHy_uu.coeff(f)
    Coef_HPHy_uu[2, 0] = HPHy_uu.coeff(g)
    Coef_HPHy_uu[3, 0] = HPHy_uu.coeff(h)
    Coef_HPHy_uu[4, 0] = HPHy_uu.coeff(n_i)
    Coef_HPHy_uu[5, 0] = HPHy_uu.coeff(n_j)

    Mat_Coef_HPHy_uu = sp.Matrix([
        [Coef_HPHy_uu[4, 0], Coef_HPHy_uu[1, 0], o, Coef_HPHy_uu[3, 0]],
        [Coef_HPHy_uu[1, 0], Coef_HPHy_uu[5, 0], Coef_HPHy_uu[3, 0], o],
        [o, Coef_HPHy_uu[0, 0], Coef_HPHy_uu[5, 0], Coef_HPHy_uu[2, 0]],
        [Coef_HPHy_uu[0, 0], o, Coef_HPHy_uu[2, 0], Coef_HPHy_uu[4, 0]]
    ])

    HPHz_uu = sp.Integer(0)

    i = 0
    while i < 3:
        j = 0
        while j < 3:
            HPHz_uu = HPHz_uu + Hz_uu[i, j] * HPbosons[i, j]
            j += 1
        i += 1

    HPHz_uu = HPHz_uu.simplify()

    Coef_HPHz_uu = sp.Matrix([o, o, o, o, o, o])
    Coef_HPHz_uu[0, 0] = HPHz_uu.coeff(e)
    Coef_HPHz_uu[1, 0] = HPHz_uu.coeff(f)
    Coef_HPHz_uu[2, 0] = HPHz_uu.coeff(g)
    Coef_HPHz_uu[3, 0] = HPHz_uu.coeff(h)
    Coef_HPHz_uu[4, 0] = HPHz_uu.coeff(n_i)
    Coef_HPHz_uu[5, 0] = HPHz_uu.coeff(n_j)

    Mat_Coef_HPHz_uu = sp.Matrix([
        [Coef_HPHz_uu[4, 0], Coef_HPHz_uu[1, 0], o, Coef_HPHz_uu[3, 0]],
        [Coef_HPHz_uu[1, 0], Coef_HPHz_uu[5, 0], Coef_HPHz_uu[3, 0], o],
        [o, Coef_HPHz_uu[0, 0], Coef_HPHz_uu[5, 0], Coef_HPHz_uu[2, 0]],
        [Coef_HPHz_uu[0, 0], o, Coef_HPHz_uu[2, 0], Coef_HPHz_uu[4, 0]]
    ])

    # Fourier transform
    deltaX = a / sp.sqrt(3) * sp.Matrix([-sp.sqrt(3) / 2, 1 / sp.Integer(2)])
    deltaY = a / sp.sqrt(3) * sp.Matrix([sp.Integer(0), sp.Integer(-1)])
    deltaZ = a / sp.sqrt(3) * sp.Matrix([sp.sqrt(3) / 2, 1 / sp.Integer(2)])

    global k1
    global k2

    k1 = sp.Symbol("k1", real=True)
    k2 = sp.Symbol("k2", real=True)
    k = sp.Matrix([k1, k2])

    sx = sp.exp(I * k.dot(deltaX))
    sy = sp.exp(I * k.dot(deltaY))
    sz = sp.exp(I * k.dot(deltaY))
    tx = sp.exp(-I * k.dot(deltaX))
    ty = sp.exp(-I * k.dot(deltaY))
    tz = sp.exp(-I * k.dot(deltaZ))

    FT_Coef_X = sp.Matrix([[ii, sx, o, sx], [tx, ii, tx, o], [o, sx, ii, sx], [tx, o,
                                                                               tx, ii]])
    FT_Coef_Y = sp.Matrix([[ii, sy, o, sy], [ty, ii, ty, o], [o, sy, ii, sy], [ty, o,
                                                                               ty, ii]])
    FT_Coef_Z = sp.Matrix([[ii, sz, o, sz], [tz, ii, tz, o], [o, sz, ii, sz], [tz, o,
                                                                               tz, ii]])
    HlswTwoSub_X_uu = sp.Matrix([[o, o, o, o], [o, o, o, o], [o, o, o, o], [o, o, o, o]])

    i = 0
    while i < 4:
        j = 0
        while j < 4:
            HlswTwoSub_X_uu[i, j] = Mat_Coef_HPHx_uu[i, j] * FT_Coef_X[i, j]
            j += 1
        i += 1

    HlswTwoSub_Y_uu = sp.Matrix([[o, o, o, o], [o, o, o, o], [o, o, o, o], [o, o, o, o]])

    i = 0
    while i < 4:
        j = 0
        while j < 4:
            HlswTwoSub_Y_uu[i, j] = Mat_Coef_HPHy_uu[i, j] * FT_Coef_Y[i, j]
            j += 1
        i += 1

    HlswTwoSub_Z_uu = sp.Matrix([[o, o, o, o], [o, o, o, o], [o, o, o, o], [o, o, o, o]])

    i = 0
    while i < 4:
        j = 0
        while j < 4:
            HlswTwoSub_Z_uu[i, j] = Mat_Coef_HPHz_uu[i, j] * FT_Coef_Z[i, j]
            j += 1
        i += 1

    # After using magnon transformation
    magfield = h * sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Assemble
    HlswFourSub_FM = 1 / 2 * (HlswTwoSub_X_uu + HlswTwoSub_Y_uu + HlswTwoSub_Z_uu) + magfield

    # coupling energy scale in unit meV
    # length scale in unit angstrom
    # Here we use Sear's paprameter, for example:

    # a=1. leave k1 and k2 arbitrary
    # add_strg: add coupling strength

    HlswFourSub_add_strg = HlswFourSub_FM.subs([(J, strg[0]), (K, strg[1]), (Gamma, strg[2]), (gamma, strg[3]), (h, strg[4]), (a, 1)])
    return HlswFourSub_add_strg

def bdg(hlsw, kx, ky):
    # Bogoliubov transformation
    sigma_3 = np.diag([1, 1, -1, -1])

    global k1
    global k2

    k1 = sp.Symbol("k1", real=True)
    k2 = sp.Symbol("k2", real=True)

    H_pure_num = sp.matrices.dense.matrix2numpy(hlsw.subs([(k1, kx), (k2, ky)])).astype(complex)
    H_BdG = np.matmul(sigma_3, H_pure_num)
    eigval_k,eigvec_k = (LA.eig(H_BdG))

    idx = np.real(eigval_k).argsort()[::-1]
    eigval_k = np.real(eigval_k[idx])
    eigvec_k = eigvec_k[:, idx]
    return eigval_k, eigvec_k

def band_high_sym(hlsw, strg):
    eng_all = np.array([])
    dk = 2 * np.pi / 20
    fig, ax = spp(facecolor='0.95')

    # a = 1
    bb = 4 * np.pi / np.sqrt(3)

    # X to K
    # step length and number of steps:
    n = bb / (2 * np.sqrt(3)) / dk
    k1_init = -np.sqrt(3) / 2 * bb
    k2_init = 0

    # denote the high symmetry point
    fig.gca().axvline(x=0, color='b', linestyle='dashed')
    x_ticks = np.array([0])

    i = 0
    while i < n:
        eng, vec = bdg(hlsw,k1_init + i * dk, k2_init)
        eng_all = np.append(eng_all, eng, axis=0)
        i += 1

    # K to gamma
    # step length and number of steps:
    n = bb / np.sqrt(3) / dk

    k1_init = -1 / np.sqrt(3) * bb
    k2_init = 0

    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.size, color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.size], axis=0)

    i = 0
    while i < n:
        eng, vec = bdg(hlsw,  k1_init + i * dk, k2_init)
        eng_all = np.append(eng_all, eng, axis=0)
        i += 1

    # gamma to Y
    # step length and number of steps:
    n = bb / 2 / dk

    k1_init = 0
    k2_init = 0

    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.size, color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.size], axis=0)

    i = 0
    while i < n:
        eng, vec = bdg(hlsw, k1_init, k2_init + i * dk)
        eng_all = np.append(eng_all, eng, axis=0)
        i += 1

    # Y to gamma'
    # step length and number of steps:
    n = np.sqrt(3) * bb / 2 / dk

    k1_init = 0
    k2_init = bb / 2

    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.size, color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.size], axis=0)

    i = 0
    while i < n:
        eng, vec = bdg(hlsw, k1_init + i * dk, k2_init)
        eng_all = np.append(eng_all, eng, axis=0)
        i += 1

    # gamma' to M
    # step length and number of steps:
    n = bb / 2 / dk

    k1_init = np.sqrt(3) * bb / 2
    k2_init = bb / 2

    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.size, color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.size], axis=0)

    i = 0
    while i < n:
        eng, vec = bdg(hlsw, k1_init + i * dk * (-np.sqrt(3) / 2), k2_init + i * dk * (-1 / 2))
        eng_all = np.append(eng_all, eng, axis=0)
        i += 1

    # M to gamma
    # step length and number of steps:
    n = bb / 2 / dk

    k1_init = np.sqrt(3) * bb / 4
    k2_init = bb / 4

    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.size, color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.size], axis=0)

    i = 0
    while i < n:
        eng, vec = bdg(hlsw, k1_init + i * dk * (-np.sqrt(3) / 2), k2_init + i * dk * (-1 / 2))
        eng_all = np.append(eng_all, eng, axis=0)
        i += 1

    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.size - 1, color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.size - 1], axis=0)

    x = np.linspace(0, eng_all.size, num=eng_all.size, endpoint=False, retstep=False, dtype=None, axis=0)

    fig.gca().set_title("Band structure along high symmetry points")
    fig.gca().set_xlabel("k")
    fig.gca().set_ylabel("$ε_{nk}\ meV$")
    fig.gca().set_xticks(x_ticks)
    fig.gca().set_xticklabels(['X', 'K', '$\Gamma$', 'Y', "$\Gamma$'", 'M', '$\Gamma$'])

    fig.gca().scatter(x, eng_all, color='k', s=1.5)
    fig.gca().set_ybound(lower=None, upper=7)
    fig.gca().text(0, 6, f"$[J,K,\Gamma,\Gamma ', h]$ = [{strg[0]}, {strg[1]}, {strg[2]}, {strg[3]}, {strg[4]}]",
                   fontsize=10, bbox={'facecolor': '0.8', 'pad': 2})
    fig.savefig('band.png')

def berry_curv(hlsw, mu, nu, kx, ky):
    sigma_3 = np.diag([1, 1, -1, -1])

    global k1
    global k2

    k1 = sp.Symbol("k1", real=True)
    k2 = sp.Symbol("k2", real=True)

    val_k, vec_k = bdg(hlsw, kx, ky)
    vec_k_dag = np.transpose(np.conjugate(vec_k))

    Hlsw_add_strg_dkx = sp.diff(hlsw, k1)
    Hlsw_add_strg_dky = sp.diff(hlsw, k2)

    Hdx_pure_num = sp.matrices.dense.matrix2numpy(Hlsw_add_strg_dkx.subs([(k1,kx),(k2,ky)])).astype(complex)
    Hdy_pure_num = sp.matrices.dense.matrix2numpy(Hlsw_add_strg_dky.subs([(k1,kx),(k2,ky)])).astype(complex)

    mat_x = np.matmul(sigma_3, np.matmul(vec_k_dag, np.matmul(Hdx_pure_num, vec_k)))
    mat_y = np.matmul(sigma_3, np.matmul(vec_k_dag, np.matmul(Hdy_pure_num, vec_k)))

    mat = [mat_x, mat_y]
    # Calculate Berry curvature Ω𝑛𝑘
    n = 0
    mat_Omega_nk = np.array([])

    while n < 2:
        m = 0
        Omega_nk = 0
        while m < 4:
            if n != m:
                Omega_nk = Omega_nk + mat[mu][n, m] * mat[nu][m, n] / np.square(val_k[n] - val_k[m])
            else:
                pass
            m += 1
        mat_Omega_nk = np.append(mat_Omega_nk, Omega_nk)
        n += 1

    return -2*np.imag(mat_Omega_nk)

def berry_curv_graph(hlsw,muu,nuu, n, rslnx, rslny):
    x = np.linspace(-np.pi, np.pi, rslnx)
    y = np.linspace(-np.pi, np.pi, rslny)
    x, y = np.meshgrid(x, y)
    ny, nx = np.shape(x)

    z = np.zeros((ny, nx))

    i = 0
    while i < ny:
        j = 0
        while j < nx:
            z[i, j] = z[i, j] + berry_curv(hlsw, 1, 0, x[i, j], y[i, j])[n]
            # print(y[i,j],x[i,j])
            j += 1
        i += 1

    fig, ax = spp(facecolor='0.95')
    pcm = ax.contourf(x, y, z)
    ax.set_title(f"Berry curvature $\Omega$ of the {n+1}th band, $[μ,ν]$=[{muu},{nuu}]")
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    fig.colorbar(pcm, ax=ax)

    fig.savefig('berry_curv.png')

def chern(hlsw, muu, nuu, rsln):
    fig, ax = spp(facecolor='0.95')

    bb = 4 * np.pi / np.sqrt(3)
    sum_k = np.array([0, 0])

    # dT = 0.1
    # T_init = 1.5
    # sum_k_varT = np.array([])

    # lower part
    ky_init = -bb / 2
    rsln_y = rsln
    dk = bb / 2 / rsln_y

    i = 0
    while i < rsln_y:
        kyy = ky_init + dk * i
        kx_init = -(kyy + bb) / np.sqrt(3)
        rsln_x = 2*(kyy + bb) / np.sqrt(3)/dk
        j = 0
        while j < rsln_x+1:
            kxx = kx_init + dk * j

            # sum over n=1~N
            mat_omega_nk = berry_curv(hlsw, muu, nuu, kxx, kyy)
            sum_k = sum_k + dk*dk * mat_omega_nk

            fig.gca().scatter(kxx,kyy)
            j += 1
        i += 1

        # -------------------------------------------------------

    # upper part
    ky_init = 0

    i = 0
    while i < rsln_y:
        kyy = ky_init + dk * i
        kx_init = -(bb-kyy) / np.sqrt(3)
        rsln_x = 2*(bb-kyy) /np.sqrt(3)/dk
        j = 0
        while j < rsln_x+1:
            kxx = kx_init + dk * j

            # sum over n=1~N
            mat_omega_nk = berry_curv(hlsw, muu, nuu, kxx, kyy)
            sum_k = sum_k + dk*dk * mat_omega_nk

            fig.gca().scatter(kxx, kyy)
            j += 1
        i += 1

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("1st BZ with Chern number")
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    # fig.colorbar(pcm, ax=ax)

    fig.savefig('1BZ_with_chern.png')

    return -sum_k/(2*np.pi)

def c2(hlsw, kx,ky, T):
    I = sp.I
    rho = sp.Symbol("ρ_{nk}", real=True)

    # complete C2(x) function
    C2 = (1 + rho) * sp.Pow(sp.log((1 + rho) / rho, sp.E), 2) - sp.Pow(sp.log(rho, sp.E), 2) + 2 * (
                np.pi * np.pi / 12 + sp.log(rho, sp.E) * sp.log((rho + 1), sp.E))

    val, vec = bdg(hlsw, kx, ky)
    delta = val / (0.0862 * T)

    c2=np.array([])
    rho_null = 1 / (np.exp(delta) - 1)

    i=0
    while i < 2:
        data = C2.subs([(rho, rho_null[i])])
        c2 = np.append(c2, data)
        i+=1

    return c2

def unnormalized_heat_conductivity(hlsw, muu, nuu, T, rsln):

    bb = 4*np.pi/np.sqrt(3)
    sum_k = np.array([0,0])

    # dT = 0.1
    # T_init = 1.5
    # sum_k_varT = np.array([])

    # lower part
    ky_init = -bb / 2
    rsln_y = rsln
    dk = bb / 2 / rsln_y

    i = 0
    while i < rsln_y:
        kyy = ky_init + dk * i
        kx_init = -(kyy + bb) / np.sqrt(3)
        rsln_x = (kyy + bb) / dk
        j = 0
        while j < rsln_x:
            kxx = kx_init + dk * j

            c2_nk = c2(hlsw,kxx,kyy,T)
            mat_omega_nk = berry_curv(hlsw, muu, nuu, kxx, kyy)


            # sum over n=1~N
            sum_k = sum_k + c2_nk*mat_omega_nk

            j += 1
        i += 1

        # -------------------------------------------------------

    # upper part
    ky_init = 0
    rsln_y = rsln
    dk = bb / 2 / rsln_y

    i = 0
    while i < rsln_y:
        kyy = ky_init + dk * i
        kx_init = (kyy - bb) / np.sqrt(3)
        rsln_x = -(kyy - bb) / dk
        j = 0
        while j < rsln_x:
            kxx = kx_init + dk * j

            c2_nk = c2(hlsw, kxx, kyy, T)
            mat_omega_nk = berry_curv(hlsw, muu, nuu, kxx, kyy)

            # sum over n=1~N
            sum_k = sum_k + c2_nk * mat_omega_nk

            j += 1
        i += 1

    kappa = sum_k*T
    return sum_k

