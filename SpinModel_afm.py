import sympy as sp
import numpy as np
from matplotlib.pyplot import subplots as spp
from numpy import linalg as LA
from sympy.physics.matrices import msigma

def distri224(H, Hsub, a, b):
    lat = {'0': a, '1': b}
    l=0
    while l<2:
        ll=0
        while ll<2:
            i=0
            while i<2:
                j=0
                while j<2:
                    H[lat[f'{l}']+i*4,lat[f'{ll}']+j*4] = H[lat[f'{l}']+i*4,lat[f'{ll}']+j*4] + Hsub[0+i*2+l,0+j*2+ll]
                    j+=1
                i+=1
            ll+=1
        l+=1

def afm_spin_model(strg):
    # afm is meaningless variable!
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
# up-up couplings
    theta =0
    phi = 0

    V_i = sp.symbols("V1:4")
    V1 = sp.Matrix([1, 0, 0])
    V2 = sp.Matrix([0, 1, 0])
    V3 = sp.Matrix([0, 0, 1])

    Hberg_uu = V1 * sp.Transpose(V1) + V2 * sp.Transpose(V2) + V3 * sp.Transpose(V3)

    IsingXX_uu = V1 * sp.Transpose(V1)
    IsingZZ_uu = V3 * sp.Transpose(V3)

    XY_uu = V1 * sp.Transpose(V2)
    XZ_uu = V1 * sp.Transpose(V3)
    YX_uu = V2 * sp.Transpose(V1)
    YZ_uu = V2 * sp.Transpose(V3)
    ZX_uu = V3 * sp.Transpose(V1)
    ZY_uu = V3 * sp.Transpose(V2)

    Hx_uu = 2 * K * IsingXX_uu + J * Hberg_uu + Gamma * (YZ_uu + ZY_uu) + gamma * (YX_uu + ZX_uu + XY_uu + XZ_uu)
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

    HPHx_uu = HPHx_uu.expand()

    Coef_HPHx_uu = sp.Matrix([o, o, o, o, o, o])
    Coef_HPHx_uu[0, 0] = HPHx_uu.coeff(e).simplify()
    Coef_HPHx_uu[1, 0] = HPHx_uu.coeff(f).simplify()
    Coef_HPHx_uu[2, 0] = HPHx_uu.coeff(g).simplify()
    Coef_HPHx_uu[3, 0] = HPHx_uu.coeff(h).simplify()
    Coef_HPHx_uu[4, 0] = HPHx_uu.coeff(n_i).simplify()
    Coef_HPHx_uu[5, 0] = HPHx_uu.coeff(n_j).simplify()

    Mat_Coef_HPHx_uu = sp.Matrix([
        [Coef_HPHx_uu[4, 0], Coef_HPHx_uu[2, 0], o, Coef_HPHx_uu[3, 0]],
        [Coef_HPHx_uu[1, 0], Coef_HPHx_uu[5, 0], Coef_HPHx_uu[3, 0], o],
        [o, Coef_HPHx_uu[0, 0], Coef_HPHx_uu[5, 0], Coef_HPHx_uu[2, 0]],
        [Coef_HPHx_uu[0, 0], o, Coef_HPHx_uu[1, 0], Coef_HPHx_uu[4, 0]]
    ])

    HPHz_uu = sp.Integer(0)

    i = 0
    while i < 3:
        j = 0
        while j < 3:
            HPHz_uu = HPHz_uu + Hz_uu[i, j] * HPbosons[i, j]
            j += 1
        i += 1

    HPHz_uu = HPHz_uu.expand()

    Coef_HPHz_uu = sp.Matrix([o, o, o, o, o, o])
    Coef_HPHz_uu[0, 0] = HPHz_uu.coeff(e).simplify()
    Coef_HPHz_uu[1, 0] = HPHz_uu.coeff(f).simplify()
    Coef_HPHz_uu[2, 0] = HPHz_uu.coeff(g).simplify()
    Coef_HPHz_uu[3, 0] = HPHz_uu.coeff(h).simplify()
    Coef_HPHz_uu[4, 0] = HPHz_uu.coeff(n_i).simplify()
    Coef_HPHz_uu[5, 0] = HPHz_uu.coeff(n_j).simplify()

    Mat_Coef_HPHz_uu = sp.Matrix([
        [Coef_HPHz_uu[4, 0], Coef_HPHz_uu[2, 0], o, Coef_HPHz_uu[3, 0]],
        [Coef_HPHz_uu[1, 0], Coef_HPHz_uu[5, 0], Coef_HPHz_uu[3, 0], o],
        [o, Coef_HPHz_uu[0, 0], Coef_HPHz_uu[5, 0], Coef_HPHz_uu[2, 0]],
        [Coef_HPHz_uu[0, 0], o, Coef_HPHz_uu[1, 0], Coef_HPHz_uu[4, 0]]
    ])


#############################################
# down-down couplings
    Theta = np.pi-theta
    Phi = np.pi-phi
    uni = sp.Matrix([[np.cos(Theta/2),np.sin(Theta / 2)],[np.sin(Theta/2)*np.exp(1j*Phi), -np.cos(Theta / 2) * np.exp(1j * Phi)]])
    uni_dag = sp.transpose(sp.conjugate(uni))


    pxd = sp.simplify(uni_dag*msigma(1)*uni)
    pyd = sp.simplify(uni_dag*msigma(2)*uni)
    pzd = sp.simplify(uni_dag * msigma(3) * uni)

    W1 = sp.simplify(sp.Matrix([(pxd[0, 1] + pxd[1, 0]) / 2, (pxd[1, 0] - pxd[0, 1]) / (2 * I), (pxd[0, 0] - pxd[1, 1]) / 2]))
    W2 = sp.simplify(sp.Matrix([(pyd[0, 1] + pyd[1, 0]) / 2, (pyd[1, 0] - pyd[0, 1]) / (2 * I), (pyd[0, 0] - pyd[1, 1]) / 2]))
    W3 = sp.simplify(sp.Matrix([(pzd[0, 1] + pzd[1, 0]) / 2, (pzd[1, 0] - pzd[0, 1]) / (2 * I), (pzd[0, 0] - pzd[1, 1]) / 2]))

    Hberg_dd = W1 * sp.Transpose(W1) + W2 * sp.Transpose(W2) + W3 * sp.Transpose(W3)

    IsingXX_dd = W1 * sp.Transpose(W1)
    IsingZZ_dd = W3 * sp.Transpose(W3)

    XY_dd = W1 * sp.Transpose(W2)
    XZ_dd = W1 * sp.Transpose(W3)
    YX_dd = W2 * sp.Transpose(W1)
    YZ_dd = W2 * sp.Transpose(W3)
    ZX_dd = W3 * sp.Transpose(W1)
    ZY_dd = W3 * sp.Transpose(W2)

    Hx_dd = 2 * K * IsingXX_dd + J * Hberg_dd + Gamma * (YZ_dd + ZY_dd) + gamma * (YX_dd + ZX_dd + XY_dd + XZ_dd)
    Hz_dd = 2 * K * IsingZZ_dd + J * Hberg_dd + Gamma * (XY_dd + YX_dd) + gamma * (XZ_dd + YZ_dd + ZX_dd + ZY_dd)

    HPHx_dd = sp.Integer(0)

    i = 0
    while i < 3:
        j = 0
        while j < 3:
            HPHx_dd = HPHx_dd + Hx_dd[i, j] * HPbosons[i, j]
            j += 1
        i += 1

    HPHx_dd = HPHx_dd.expand()

    Coef_HPHx_dd = sp.Matrix([o, o, o, o, o, o])
    Coef_HPHx_dd[0, 0] = HPHx_dd.coeff(e).simplify()
    Coef_HPHx_dd[1, 0] = HPHx_dd.coeff(f).simplify()
    Coef_HPHx_dd[2, 0] = HPHx_dd.coeff(g).simplify()
    Coef_HPHx_dd[3, 0] = HPHx_dd.coeff(h).simplify()
    Coef_HPHx_dd[4, 0] = HPHx_dd.coeff(n_i).simplify()
    Coef_HPHx_dd[5, 0] = HPHx_dd.coeff(n_j).simplify()

    Mat_Coef_HPHx_dd = sp.Matrix([
        [Coef_HPHx_dd[4, 0], Coef_HPHx_dd[2, 0], o, Coef_HPHx_dd[3, 0]],
        [Coef_HPHx_dd[1, 0], Coef_HPHx_dd[5, 0], Coef_HPHx_dd[3, 0], o],
        [o, Coef_HPHx_dd[0, 0], Coef_HPHx_dd[5, 0], Coef_HPHx_dd[2, 0]],
        [Coef_HPHx_dd[0, 0], o, Coef_HPHx_dd[1, 0], Coef_HPHx_dd[4, 0]]
    ])

    HPHz_dd = sp.Integer(0)

    i = 0
    while i < 3:
        j = 0
        while j < 3:
            HPHz_dd = HPHz_dd + Hz_dd[i, j] * HPbosons[i, j]
            j += 1
        i += 1

    HPHz_dd = HPHz_dd.expand()

    Coef_HPHz_dd = sp.Matrix([o, o, o, o, o, o])
    Coef_HPHz_dd[0, 0] = HPHz_dd.coeff(e).simplify()
    Coef_HPHz_dd[1, 0] = HPHz_dd.coeff(f).simplify()
    Coef_HPHz_dd[2, 0] = HPHz_dd.coeff(g).simplify()
    Coef_HPHz_dd[3, 0] = HPHz_dd.coeff(h).simplify()
    Coef_HPHz_dd[4, 0] = HPHz_dd.coeff(n_i).simplify()
    Coef_HPHz_dd[5, 0] = HPHz_dd.coeff(n_j).simplify()

    Mat_Coef_HPHz_dd = sp.Matrix([
        [Coef_HPHz_dd[4, 0], Coef_HPHz_dd[2, 0], o, Coef_HPHz_dd[3, 0]],
        [Coef_HPHz_dd[1, 0], Coef_HPHz_dd[5, 0], Coef_HPHz_dd[3, 0], o],
        [o, Coef_HPHz_dd[0, 0], Coef_HPHz_dd[5, 0], Coef_HPHz_dd[2, 0]],
        [Coef_HPHz_dd[0, 0], o, Coef_HPHz_dd[1, 0], Coef_HPHz_dd[4, 0]]
    ])

#####################################################################
    # up-down couplings
    Hberg_ud = V1 * sp.Transpose(W1) + V2 * sp.Transpose(W2) + V3 * sp.Transpose(W3)

    IsingXX_ud = V1 * sp.Transpose(W1)
    IsingYY_ud = V2 * sp.Transpose(W2)
    IsingZZ_ud = V3 * sp.Transpose(W3)

    XY_ud = V1 * sp.Transpose(W2)
    XZ_ud = V1 * sp.Transpose(W3)
    YX_ud = V2 * sp.Transpose(W1)
    YZ_ud = V2 * sp.Transpose(W3)
    ZX_ud = V3 * sp.Transpose(W1)
    ZY_ud = V3 * sp.Transpose(W2)

    Hy_ud = 2 * K * IsingYY_ud + J * Hberg_ud + Gamma * (XZ_ud + ZX_ud) + gamma * (YZ_ud + YX_ud + ZY_ud + XY_ud)
    HPHy_ud = sp.Integer(0)

    i = 0
    while i < 3:
        j = 0
        while j < 3:
            HPHy_ud = HPHy_ud + Hy_ud[i, j] * HPbosons[i, j]
            j += 1
        i += 1

    HPHy_ud = HPHy_ud.expand()

    Coef_HPHy_ud = sp.Matrix([o, o, o, o, o, o])
    Coef_HPHy_ud[0, 0] = HPHy_ud.coeff(e).simplify()
    Coef_HPHy_ud[1, 0] = HPHy_ud.coeff(f).simplify()
    Coef_HPHy_ud[2, 0] = HPHy_ud.coeff(g).simplify()
    Coef_HPHy_ud[3, 0] = HPHy_ud.coeff(h).simplify()
    Coef_HPHy_ud[4, 0] = HPHy_ud.coeff(n_i).simplify()
    Coef_HPHy_ud[5, 0] = HPHy_ud.coeff(n_j).simplify()

    Mat_Coef_HPHy_ud = sp.Matrix([
        [Coef_HPHy_ud[4, 0], Coef_HPHy_ud[2, 0], o, Coef_HPHy_ud[3, 0]],
        [Coef_HPHy_ud[1, 0], Coef_HPHy_ud[5, 0], Coef_HPHy_ud[3, 0], o],
        [o, Coef_HPHy_ud[0, 0], Coef_HPHy_ud[5, 0], Coef_HPHy_ud[2, 0]],
        [Coef_HPHy_ud[0, 0], o, Coef_HPHy_ud[1, 0], Coef_HPHy_ud[4, 0]]
    ])
######################################################################
    # down-up couplings
    Hberg = W1 * sp.Transpose(V1) + W2 * sp.Transpose(V2) + W3 * sp.Transpose(V3)

    IsingXX = W1 * sp.Transpose(V1)
    IsingYY = W2 * sp.Transpose(V2)
    IsingZZ = W3 * sp.Transpose(V3)

    XY = W1 * sp.Transpose(V2)
    XZ = W1 * sp.Transpose(V3)
    YX = W2 * sp.Transpose(V1)
    YZ = W2 * sp.Transpose(V3)
    ZX = W3 * sp.Transpose(V1)
    ZY = W3 * sp.Transpose(V2)

    Hy = 2 * K * IsingYY + J * Hberg + Gamma * (XZ + ZX) + gamma * (YZ + YX + ZY + XY)
    HPHy = sp.Integer(0)

    i = 0
    while i < 3:
        j = 0
        while j < 3:
            HPHy = HPHy + Hy[i, j] * HPbosons[i, j]
            j += 1
        i += 1

    HPHy = HPHy.expand()

    Coef_HPHy = sp.Matrix([o, o, o, o, o, o])
    Coef_HPHy[0, 0] = HPHy.coeff(e).simplify()
    Coef_HPHy[1, 0] = HPHy.coeff(f).simplify()
    Coef_HPHy[2, 0] = HPHy.coeff(g).simplify()
    Coef_HPHy[3, 0] = HPHy.coeff(h).simplify()
    Coef_HPHy[4, 0] = HPHy.coeff(n_i).simplify()
    Coef_HPHy[5, 0] = HPHy.coeff(n_j).simplify()

    Mat_Coef_HPHy_du = sp.Matrix([
        [Coef_HPHy[4, 0], Coef_HPHy[2, 0], o, Coef_HPHy[3, 0]],
        [Coef_HPHy[1, 0], Coef_HPHy[5, 0], Coef_HPHy[3, 0], o],
        [o, Coef_HPHy[0, 0], Coef_HPHy[5, 0], Coef_HPHy[2, 0]],
        [Coef_HPHy[0, 0], o, Coef_HPHy[1, 0], Coef_HPHy[4, 0]]
    ])

    #####################################################################
    Hlsw = sp.zeros(8, 8)

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
    sz = sp.exp(I * k.dot(deltaZ))
    tx = sp.exp(-I * k.dot(deltaX))
    ty = sp.exp(-I * k.dot(deltaY))
    tz = sp.exp(-I * k.dot(deltaZ))

    FT_Coef_X = sp.Matrix([[ii, sx, o, sx], [tx, ii, tx, o], [o, sx, ii, sx], [tx, o,
                                                                               tx, ii]])
    FT_Coef_Y = sp.Matrix([[ii, sy, o, sy], [ty, ii, ty, o], [o, sy, ii, sy], [ty, o,
                                                                               ty, ii]])
    FT_Coef_Z = sp.Matrix([[ii, sz, o, sz], [tz, ii, tz, o], [o, sz, ii, sz], [tz, o,
                                                                               tz, ii]])

# x bonds:

    # up-up x bonds
    HlswTwoSub = sp.Matrix([[o, o, o, o], [o, o, o, o], [o, o, o, o], [o, o, o, o]])

    i = 0
    while i < 4:
        j = 0
        while j < 4:
            HlswTwoSub[i, j] = Mat_Coef_HPHx_uu[i, j] * FT_Coef_X[i, j]
            j += 1
        i += 1

    distri224(Hlsw, HlswTwoSub, 1, 0)

    HlswTwoSub = sp.Matrix([[o, o, o, o], [o, o, o, o], [o, o, o, o], [o, o, o, o]])

    # down-down x bonds
    i = 0
    while i < 4:
        j = 0
        while j < 4:
            HlswTwoSub[i, j] = Mat_Coef_HPHx_dd[i, j] * FT_Coef_X[i, j]
            j += 1
        i += 1

    distri224(Hlsw, HlswTwoSub, 3, 2)

# y bonds:

    # up-down y bonds
    HlswTwoSub = sp.Matrix([[o, o, o, o], [o, o, o, o], [o, o, o, o], [o, o, o, o]])

    i = 0
    while i < 4:
        j = 0
        while j < 4:
            HlswTwoSub[i, j] = Mat_Coef_HPHy_ud[i, j] * FT_Coef_Y[i, j]
            j += 1
        i += 1

    distri224(Hlsw, HlswTwoSub, 1, 2)

    # down-up y bonds
    HlswTwoSub = sp.Matrix([[o, o, o, o], [o, o, o, o], [o, o, o, o], [o, o, o, o]])

    i = 0
    while i < 4:
        j = 0
        while j < 4:
            HlswTwoSub[i, j] = Mat_Coef_HPHy_du[i, j] * FT_Coef_Y[i, j]
            j += 1
        i += 1

    distri224(Hlsw, HlswTwoSub, 3, 0)

# z bonds:

    # up-up z bonds
    HlswTwoSub = sp.Matrix([[o, o, o, o], [o, o, o, o], [o, o, o, o], [o, o, o, o]])

    i = 0
    while i < 4:
        j = 0
        while j < 4:
            HlswTwoSub[i, j] = Mat_Coef_HPHz_uu[i, j] * FT_Coef_Z[i, j]
            j += 1
        i += 1

    distri224(Hlsw, HlswTwoSub, 1, 0)

    # down-down z bonds
    HlswTwoSub = sp.Matrix([[o, o, o, o], [o, o, o, o], [o, o, o, o], [o, o, o, o]])

    i = 0
    while i < 4:
        j = 0
        while j < 4:
            HlswTwoSub[i, j] = Mat_Coef_HPHz_dd[i, j] * FT_Coef_Z[i, j]
            j += 1
        i += 1

    distri224(Hlsw, HlswTwoSub, 3, 2)
    ##########################################################
    # After using magnon transformation
    magfield = 1/2 * h * sp.diag(1, 1, -1, -1, 1, 1, -1, -1)
    # Assemble
    Hlsw = 1 / 2 * Hlsw + magfield
    Hlsw_add_strg = Hlsw.subs([(J, strg[0]), (K, strg[1]), (Gamma, strg[2]), (gamma, strg[3]), (h,strg[4]), (a, 1)])
    
    return Hlsw_add_strg

def bdg(hlsw, kx, ky):
    # Bogoliubov transformation
    sigma_3 = np.diag([1, 1, 1, 1, -1, -1, -1, -1])

    global k1
    global k2

    k1 = sp.Symbol("k1", real=True)
    k2 = sp.Symbol("k2", real=True)

    H_pure_num = sp.matrices.dense.matrix2numpy(hlsw.subs([(k1, kx), (k2, ky)])).astype(complex)

    H_test = H_pure_num-np.transpose(np.conjugate(H_pure_num))

    H_BdG = np.matmul(sigma_3, H_pure_num)
    eigval_k,eigvec_k = (LA.eig(H_BdG))

    idx = np.real(eigval_k).argsort()[::-1]
    eigval_k = eigval_k[idx]
    eigvec_k = eigvec_k[:, idx]
    return eigval_k, eigvec_k

def band_high_sym(hlsw, strg, rsln):
    eng_all = np.empty((0,8), float)
    dk = 2 * np.pi / rsln
    fig, ax = spp(facecolor='0.95')

    # a = 1
    bb = 4 * np.pi / np.sqrt(3)

# gamma to X
    # step length and number of steps:
    n = np.sqrt(3)*bb / 4 / dk
    k1_init = 0
    k2_init = 0

    # denote the high symmetry point
    fig.gca().axvline(x=0, color='b', linestyle='dashed')
    x_ticks = np.array([0])

    i = 0
    while i < n:
        eng, vec = bdg(hlsw,k1_init + i * dk, k2_init)
        eng_all = np.append(eng_all, [eng], axis=0)
        print(k1_init + i * dk, k2_init)
        i += 1

# X to M
    # step length and number of steps:
    n = bb / 4 / dk

    k1_init = np.sqrt(3)*bb / 4
    k2_init = 0

    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.shape[0], color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.shape[0]], axis=0)

    i = 0
    while i < n:
        eng, vec = bdg(hlsw,  k1_init, k2_init - i * dk)
        eng_all = np.append(eng_all, [eng], axis=0)
        print(k1_init, k2_init - i * dk)
        i += 1

    # M to Y
    # step length and number of steps:
    n = np.sqrt(3)*bb / 4 / dk

    k1_init = np.sqrt(3)*bb/4
    k2_init = bb/4

    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.shape[0], color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.shape[0]], axis=0)

    i = 0
    while i < n:
        eng, vec = bdg(hlsw, k1_init - i * dk, k2_init)
        eng_all = np.append(eng_all, [eng], axis=0)
        print(k1_init - i * dk, k2_init)
        i += 1

    # Y to gamma
    # step length and number of steps:
    n = bb / 4 / dk

    k1_init = 0
    k2_init = bb / 4

    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.shape[0], color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.shape[0]], axis=0)

    i = 0
    while i < n:
        eng, vec = bdg(hlsw, k1_init, k2_init - i * dk)
        eng_all = np.append(eng_all, [eng], axis=0)
        print(k1_init, k2_init - i * dk)
        i += 1

    # gamma to M
    # step length and number of steps:
    n = bb / 2 / dk

    k1_init = 0
    k2_init = 0

    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.shape[0], color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.shape[0]], axis=0)

    i = 0
    while i < n:
        eng, vec = bdg(hlsw, k1_init + i * dk * (np.sqrt(3) / 2), k2_init + i * dk * (1 / 2))
        eng_all = np.append(eng_all, [eng], axis=0)
        print( k1_init + i * dk * (np.sqrt(3) / 2), k2_init + i * dk * (1 / 2))
        i += 1
    
    # denote the high symmetry point
    fig.gca().axvline(x=eng_all.shape[0] - 1, color='b', linestyle='dashed')
    x_ticks = np.append(x_ticks, [eng_all.shape[0] - 1], axis=0)

    num = int(eng_all.size/8)
    x = np.linspace(0, num, num, endpoint=False, retstep=False, dtype=None, axis=0)

    fig.gca().set_title("Band structure along high symmetry points")
    fig.gca().set_xlabel("k")
    fig.gca().set_ylabel("$ε_{nk}\ meV$")
    fig.gca().set_xticks(x_ticks)
    fig.gca().set_xticklabels(['$\Gamma$', 'X', 'M', 'Y', "$\Gamma$", 'M'])

    print(eng_all)
    eng_all = np.reshape(eng_all,(-1,8))
    print(np.real(eng_all).max(), np.real(eng_all).min())

    i = 0
    while i<4:
        fig.gca().plot(x, np.real(eng_all[:, i]), color='k')
        i+=1

    fig.gca().set_ybound(lower=None, upper=np.real(eng_all).max()+5.5)
    fig.gca().text(0, np.real(eng_all).max()+3, f"$[J,K,\Gamma,\Gamma ', h]$ = [{strg[0]}, {strg[1]}, {strg[2]}, {strg[3]}, {strg[4]}]",
                   fontsize=10, bbox={'facecolor': '0.8', 'pad': 2})
    fig.savefig('band.png')

def berry_curv(hlsw, mu, nu, kx, ky):
    sigma_3 = np.diag([1, 1, 1, 1, -1, -1, -1, -1])

    global k1
    global k2

    k1 = sp.Symbol("k1", real=True)
    k2 = sp.Symbol("k2", real=True)

    val_k, vec_k = bdg(hlsw, kx, ky)
    val_k = np.real(val_k)
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

    while n < 4:
        m = 0
        Omega_nk = 0
        while m < 8:
            if n != m:
                Omega_nk = Omega_nk + mat[mu][n, m] * mat[nu][m, n] / np.square(val_k[n] - val_k[m])
            else:
                pass
            m += 1
        mat_Omega_nk = np.append(mat_Omega_nk, Omega_nk)
        n += 1

    return -2*np.imag(mat_Omega_nk)


# The band with index [0] is in fact the band with highest energy!!!
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
            z[i, j] = z[i, j] + berry_curv(hlsw, muu, nuu, x[i, j], y[i, j])[3-n]
            # print(y[i,j],x[i,j])
            j += 1
        i += 1

    fig, ax = spp(facecolor='0.95')
    pcm = ax.contourf(x, y, z)
    ax.set_title(f"Berry curvature $\Omega$ of the {4-n}th band, $[μ,ν]$=[{muu},{nuu}]")
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    fig.colorbar(pcm, ax=ax)

    fig.savefig('afm_berry_curv.png')

def chern(hlsw, muu, nuu, rsln):
    fig, ax = spp(facecolor='0.95')
    N = 0
    bb = 4 * np.pi / np.sqrt(3)
    sum_k = np.array([0, 0, 0, 0])

    # dT = 0.1
    # T_init = 1.5
    # sum_k_varT = np.array([])

    # lower part
    ky_init = -bb / 4
    rsln_y = rsln
    dk = bb / 2 / rsln_y

    i = 0
    while i < rsln_y:
        kyy = ky_init + dk * i
        kx_init = -np.sqrt(3)*bb/4
        rsln_x = np.sqrt(3)*bb/2/dk
        j = 0
        while j < rsln_x:
            kxx = kx_init + dk * j

            # sum over n=1~N
            mat_omega_nk = berry_curv(hlsw, muu, nuu, kxx, kyy)
            sum_k = sum_k + dk*dk * mat_omega_nk
            N+=1
            fig.gca().scatter(kxx,kyy)
            j += 1
        i += 1

        # -------------------------------------------------------

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("1st BZ with Chern number")
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    # fig.colorbar(pcm, ax=ax)

    fig.savefig('1BZ_with_chern.png')

    return -sum_k/(2*np.pi), N

def c2(hlsw, kx,ky, T):
    I = sp.I
    rho = sp.Symbol("ρ_{nk}", real=True)

    # complete C2(x) function
    C2 = (1 + rho) * sp.Pow(sp.log((1 + rho) / rho, sp.E), 2) - sp.Pow(sp.log(rho, sp.E), 2) + 2 * (
                np.pi * np.pi / 12 + sp.log(rho, sp.E) * sp.log((rho + 1), sp.E))

    val, vec = bdg(hlsw, kx, ky)
    delta = np.real(val) / (0.0862 * T)

    print(val)
    c2=np.array([])
    rho_null = 1 / (np.exp(delta) - 1)

    i=0
    while i < 4:
        data = C2.subs([(rho, rho_null[i])])
        c2 = np.append(c2, data)
        i+=1

    return c2


def unnormalized_heat_conductivity(hlsw, muu, nuu, T, rsln):
    bb = 4 * np.pi / np.sqrt(3)
    sum_k = np.array([0, 0, 0, 0])
    #of unit cells!
    N = 0

    # dT = 0.1
    # T_init = 1.5
    # sum_k_varT = np.array([])

    ky_init = -bb / 4
    rsln_y = rsln
    dk = bb / 2 / rsln_y

    i = 0
    while i < rsln_y:
        kyy = ky_init + dk * i
        kx_init = -np.sqrt(3) * bb / 4
        rsln_x = np.sqrt(3) * bb / 2 / dk
        j = 0
        while j < rsln_x:
            kxx = kx_init + dk * j
            c2_nk = c2(hlsw,kxx,kyy,T)
            mat_omega_nk = berry_curv(hlsw, muu, nuu, kxx, kyy)
            # data = c2_nk*mat_omega_nk
            print(c2_nk)
            #if np.imag(np.sum(data))!=0:
                #data=0
            #print(f"somehow we have imaginary part at k = ({kxx}),{kyy}")
            # sum over n=1~N
            #sum_k = sum_k + data
            N += 1
            j += 1
        i += 1

        # -------------------------------------------------------

    kappa = -sum_k*T/N
    return kappa, N
