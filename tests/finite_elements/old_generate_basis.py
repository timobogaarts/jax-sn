import sympy
import numpy as np
from scipy.interpolate import lagrange
from sympy import Number
from IPython.display import Math, display


### Scripts written for Python based on:
### [1] Hesthaven, Jan S., and Tim Warburton. Nodal discontinuous Galerkin methods: algorithms, analysis, and applications. Springer Science & Business Media, 2007.
###
# ===============================================================================================================================================================
#                                                      Common Functions                                                                                         #
# ===============================================================================================================================================================

def bar(poly, symbols):
    ''' 
    Puts a bar around the symbols in the expression
        Useful for substituting symbol by symbol.
            (e.g., subsituting r*s with r -> s and s->t naively yields t**2 while it should be s * t.
            Therefore, first overline and then substitute.)
    '''
    poly2 = poly
    for i in symbols:
        ib = sympy.symbols("\\overline{"+str(i)+"}")
        poly2 = poly2.subs({i : ib})
    return poly2

def JacobiGQ(alpha, beta, N):
    '''
    Computes the Nth order gauss quadrature points associated with the Jacobi polynomial (alpha,beta).
    
    Directly rewritten from  [1]
    '''

    p = np.zeros(N+1)
    if(N==0):
        p[0] = (alpha - beta) / (alpha + beta + 2)
        return p
    
    J = np.zeros(N+1)

    h1 = 2 * np.arange(0,N+1) + alpha + beta
    
    oneton = np.arange(1,N+1)    
    
    J = np.diag(-0.5 * (alpha**2 - beta**2)/(h1 + 2) / h1) + np.diag(2.0 / (h1[0:-1] + 2) * np.sqrt( oneton * (oneton + alpha + beta) * (oneton + alpha) * (oneton + beta)/ (h1[0:-1] + 1) / (h1[0:-1] + 3)), 1)
    
    J = J + np.transpose(J)
    
    
    val, vec = np.linalg.eig(J)
    return np.sort(val)


def JacobiGL(N, alpha, beta):
    '''
    Computes the Nth order Gauss Lobatto quadrature points. 
    
    Directly rewritten from [1]
    '''
    p = np.zeros(N+1)

    if N == 0:
        p[0] = 0.0
        return p    
    elif N==1:
        p[0] = -1.0
        p[1] = 1.0
        return p
    else:
        p = np.concatenate([[-1], JacobiGQ(alpha + 1, beta +1, N-2),[1]])
        return p
def weight(n, alpha, beta):
    '''
    Weight for the Jacobi polynomial such that it is orthonormal instead of orthogonal
    '''
    return sympy.sqrt(2**(alpha + beta  + 1) * sympy.gamma(n + alpha + 1) * sympy.gamma(n+ beta + 1) / sympy.gamma(n + alpha + beta + 1) / sympy.factorial(n)) / sympy.sqrt( 2 * n + alpha + beta + 1 )


def jacobi_poly(n,alpha, beta, symbol): 
    '''
        Returns a sympy orthogonal Jacobi polynomial.
    '''
    return sympy.polys.orthopolys.jacobi_poly(n, alpha, beta, symbol)/ weight(n,alpha,beta)
 

def w(n_order):
    '''
        Computes the 1D lagrangian polynomial function from Gauss-Lobatto points to equidistant points
    '''
    lgl = JacobiGL(n_order,0,0)
    req = np.linspace(-1, 1, n_order + 1)
    n_poly = n_order + 1    

    wpoly2 = lagrange(req, lgl -req)    
    
    return wpoly2


def round2(eq):
    '''
    Rounds away extremely small floating point numbers in SymPy expressions. 
        Can be used to simplify matrices and basis functions. It does not modify or round non-almost-zero factors.
    '''
    result = eq
    eps = 1e-13
    for a in sympy.preorder_traversal(eq):        
        if isinstance(a, sympy.core.numbers.Float):            
            if np.abs(a) < eps:                
                result = result.subs(a, 0.0)                
    return result


def pprintm(mat):
    '''
    Pretty-prints for Jupyter notebook
    '''
    display(Math(sympy.latex(mat)))


def pprintl(list):
    '''
    Pretty-prints for Jupyter notebook
    '''
    for i in list:
        display(Math(sympy.latex(i)))

def proper(mat):
    '''
    Rounds away extremely small floating point numbers in numpy matrices expressions. 
        Can be used to simplify matrices. It does not modify or round non-almost-zero factors.
    '''
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if(np.abs(mat[i,j]) < 1e-11):
                mat[i,j] =0.0

    return mat

# ===============================================================================================================================================================
#                                                                          1D Functions                                                                         #
# ===============================================================================================================================================================

def generate_jacobi_base_0D(N):
    r = sympy.symbols("r")
    return [jacobi_poly(0, 0,0,r) * sympy.sqrt(2)] # It should be normalised when *evaluating* it in 0D.
def Nodes1D(N):
    '''
    1D Nodes of order N
    '''
    return JacobiGL(N, 0,0)

def generate_jacobi_base_1D(N):
    '''
    Generates a 1D ortonormal basis of order order N
    '''
    basis = []
    r = sympy.symbols("r")
    for i in range(N + 1):        
        basis.append(jacobi_poly(i, 0,0,r))
    return basis


def generate_basis_1D(ri, N):
    '''
    Generates the 1D Lagrangian basis functions of order N given node points ri
    '''
    def generate_monomial_basis_1D(N):
        order_list = []
        r = sympy.symbols("r")
        for k in range(0,N + 1):
            order_list.append(r**k)
        return order_list
    
    ri
    
    basis       = generate_monomial_basis_1D(N)      
    

    def assemble_matrix(node_points, monomial_basis):
        if(len(node_points) != len(monomial_basis)):
            raise Exception("Node point size does not match monomial basis size!")
        N = len(node_points)
        Vandermonde =np.zeros((N,N))

        for i in range(0,N):
            for j in range(0,N):
                Vandermonde[i,j] = monomial_basis[j].subs({"r" :node_points[i]}).evalf()
    
        return Vandermonde
    
    hij = np.linalg.inv(assemble_matrix(ri, basis))
    
    basis_functions = []


    for i in range(hij.shape[0]):
        basis_functions.append(sum([basis[j] * hij[j,i] for j in range(hij.shape[0])]))
    return basis_functions


# ===============================================================================================================================================================
#                                                                          2D Functions                                                                         #
# ===============================================================================================================================================================




def equidistant_baryic_2D(N):
    '''
    Equidistant coordinates for the 2D equiliteral triangle.
    
    Directly rewritten from [1]
    '''

    ijpairs =[]
    if N == 0:
        return np.array()
    
    for i in range(N + 1):
        for j in range(N + 1 - i):            
                ijpairs.append((i,j))

    return np.array([(float(i) / N , 1- float(i) / N- float(j)/ N, float(j) / N ) for i,j in ijpairs])


def Wtilde(values, wfunc):
    '''
    Modified warping function with 1/(1-r**2)

    Directly rewritten from [1]
    '''
    sf = np.ones(values.shape[0])
    slicev = np.abs(values) < 1.0
    sf[slicev] = 1/(1- values[slicev]**2)
    
    return wfunc(values) *  sf
def generate_jacobi_base_2D(N):
    '''
    Generates a 2D orthonormal of basis N

    From [1]
    '''
    r, s = sympy.symbols('r s')
    bases = []
    for i in range(N + 1):
        for j in range(N + 1):
            if j  + i  <= N:
                a = 2 * (1 + r) / ( 1- s) - 1
                b = s
                
                bases.append(sympy.cancel(sympy.simplify(sympy.sqrt(2) * jacobi_poly(i, 0, 0, a) * jacobi_poly(j, 2 * i + 1, 0, b)  * (1- b)**i)))
    return bases



def Nodes2D(N):
    '''
    2D nodes of order N

    Directly rewritten from [1]
    '''
    if N == 0:
        return np.array([- 1/ 3]), np.array([-1 /3])
    wfunc = w(N)
    alpha = 0.0
    eq_values = equidistant_baryic_2D(N)
    L1 = eq_values[:,0]
    L2 = eq_values[:,1]
    L3 = eq_values[:,2]
    
    
    warp1 = 4 * L3 * L2 * Wtilde(L3 - L2, wfunc)
    warp2 = 4 * L3 * L1 * Wtilde(L1 - L3, wfunc)
    warp3 = 4 * L2 * L1 * Wtilde(L2 - L1, wfunc)
    
    x = - L2 + L3
    y = (-L2 - L3 + 2 * L1) / np.sqrt(3.0)
    
    x = x + warp1 + np.cos( 2 * np.pi / 3.0) * warp2 + np.cos(4 * np.pi / 3) * warp3
    y = y + np.sin(2 * np.pi / 3)* warp2 + np.sin(4 * np.pi / 3) * warp3

    L1_n = (np.sqrt(3) * y + 1) / 3
    L2_n = (-3 * x - np.sqrt(3) * y + 2.0) /6
    L3_n = (3 * x  - np.sqrt(3) * y + 2.0) /6
    r = - L2_n + L3_n - L1_n
    s = - L2_n - L3_n + L1_n
    return r,s



def generate_basis_2D(ri, si, N):
    '''
    Generates the Lagrangian basis of order N in 2D given the node points ri, si.
    '''
    def generate_multivariable_monomial_basis(N):
        r, s = sympy.symbols("r s")
    
        order_list = []
        for k in range(0,N + 1):
            for l in range(0,N + 1):
                if(k+l <= N):
                    order_list.append(r**k* s**l)
            
        
        return order_list
    
    basis = generate_multivariable_monomial_basis(N)
    
    def assemble_matrix(ri, si, monomial_basis):
        
        N = ri.shape[0]
        Vandermonde =np.zeros((N,N))
    
        for i in range(0,N):
            for j in range(0,N):
                Vandermonde[i,j] = monomial_basis[j].subs({"r" : ri[i], "s" : si[i]}).evalf()
        
        return Vandermonde
        
    hij = np.linalg.inv(assemble_matrix(ri, si, basis))
    basis_functions = []
    
    for i in range(hij.shape[0]):
    
        basis_functions.append(sum([basis[j] * hij[j,i] for j in range(hij.shape[0])]))
    return basis_functions



def CreateVanderMonde(coords_rep, jacobi_functions):
    N = len(jacobi_functions)
    vandermonde = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            vandermonde[i,j] = jacobi_functions[j].subs(coords_rep[i])

    return vandermonde

# ===============================================================================================================================================================
#                                                                          3D Functions                                                                         #
# ===============================================================================================================================================================
def EquiNodes3D(p):
    '''
    Creates equi-spaced nodes in 3D given order p
    
    Directly from [1]
    '''
    N = int((p + 1) * (p + 2) * (p + 3) / 6)
    X = np.zeros(N)
    Y = np.zeros(N)
    Z = np.zeros(N)

    tot = 0

    for i in range(p + 1):
       
       for j in range(p + 1 - i):          
          for k in range(p + 1 - i - j):             
             X[tot] = - 1 + 2.0  * k / p
             Y[tot] = - 1 + 2.0  * j / p
             Z[tot] = - 1 + 2.0  * i / p
             tot +=1
    return X, Y, Z

def evalwarp3D(p, xnodes, xout):
   '''
   3D warping function

   Directly from [1]
   '''
   warp = np.zeros_like(xout)
   
   xeq = np.zeros(p + 1)
   for i in range(p + 1):
      xeq[i] = -1 + 2*(p-i)/p
   
   for i in range(p + 1):
      d = (xnodes[i]-xeq[i])
      for j in range(1, p):
         if(i !=j):
            d = d*(xout-xeq[j])/(xeq[i]-xeq[j])
            
   
      if i !=0:
         d = -d/(xeq[i] - xeq[0])

      if i != p:
         d = d/ (xeq[i]-xeq[p])

   
      warp = warp+d

   return warp

def EvalShift3D(p, pval, L1, L2, L3):
   '''
   Evaluating shifting and warping in 3D

   Directly from [1]
   '''
   gauss  = - JacobiGL(p, 0, 0)

   blend1 = L2 * L3
   blend2 = L1 * L3 
   blend3 = L1 * L2

   warpfactor1 = 4  * evalwarp3D(p, gauss, L3 - L2)
   warpfactor2 = 4  * evalwarp3D(p, gauss, L1 - L3)
   warpfactor3 = 4 *  evalwarp3D(p, gauss,  L2 - L1)


   warp1 = blend1 * warpfactor1 * (1 + (pval * L1)**2)
   
   warp2 = blend2 * warpfactor2 * (1 + (pval * L2)**2)
   warp3 = blend3 * warpfactor3 * (1 + (pval * L3)**2)

   dx = 1 * warp1 + np.cos(2 * np.pi / 3) * warp2 + np.cos(4 * np.pi / 3) * warp3
   dy = 0 * warp1 + np.sin(2 * np.pi / 3) * warp2 + np.sin(4 * np.pi / 3) * warp3

   return dx, dy



def WarpShiftFace3D(p, pval, pval2, L1, L2, L3, L4):
   '''
   Adding normals onto the faces of the tetrahedron

   Directly from [1]
   '''
   dtan1, dtan2 = EvalShift3D(p, pval, L2, L3, L4)
   warpx = dtan1
   warpy = dtan2
   return warpx, warpy


def xyztorst(X, Y, Z):
   '''
   Equilateral tetrahedron to standard tetrahedron coordinates

   Note that r here is different from [1]: the 0 vertex is at (0,0,0) in the coordinates used here.
   '''
   v1 = np.array([-1,-1/np.sqrt(3), -1/np.sqrt(6)])
   v2 = np.array([ 1,-1/np.sqrt(3), -1/np.sqrt(6)])
   v3 = np.array([ 0, 2/np.sqrt(3), -1/np.sqrt(6)])
   v4 = np.array([ 0, 0/np.sqrt(3), 3/np.sqrt(6)])

   rhs = np.vstack([X, Y, Z]) - np.outer(0.5*(v2+v3+v4-v1), np.ones((1,X.shape[0])))

   
   A = np.vstack([0.5*(v2-v1),0.5*(v3-v1),0.5*(v4-v1)]).T

   RST = np.linalg.solve(A, rhs)      
   
   r = RST[0,:]
   s = RST[1,:]
   t = RST[2,:]
   
   r = (r + 1) / 2 # modified from [1]
   s = (s + 1) / 2 # modified from [1]
   t = (t + 1) / 2 # modified from [1]
   def round_numpy(a):
      a[np.abs(a) < 1e-10] = 0.0
      return a
   return round_numpy(r), round_numpy(s), round_numpy(t)

def Nodes3D(p):
    '''
    3D nodes for order p 

    Directly from [1].
    '''
    if p == 0:
        return np.array([0.25]), np.array([0.25]), np.array([0.25])
    alpha = 0.0

    N = (p + 1) * (p + 2) * (p + 3) / 6
    tol = 1e-10

    r, s, t = EquiNodes3D(p)

    L1 = (1 + t ) / 2
    L2 = (1 + s ) / 2
    L3 = -(1 + r + s +t) / 2
    L4 = (1 + r) / 2

    v1 = np.array([-1, -1 / np.sqrt(3), -1 / np.sqrt(6)])
    v2 = np.array([ 1, -1 / np.sqrt(3), -1 / np.sqrt(6)])
    v3 = np.array([ 0,  2 / np.sqrt(3), -1 / np.sqrt(6)])
    v4 = np.array([ 0,               0,  3 / np.sqrt(6)])

    t1 = np.zeros((4, v1.shape[0]))
    t2 = np.zeros((4, v1.shape[0]))

    t1[0, :] = v2 - v1
    t1[1, :] = v2 - v1
    t1[2, :] = v3 - v2
    t1[3, :] = v3 - v1

    t2[0, :] = v3 - 0.5 * (v1 + v2) 
    t2[1, :] = v4 - 0.5 * (v1 + v2) 
    t2[2, :] = v4 - 0.5 * (v2 + v3) 
    t2[3, :] = v4 - 0.5 * (v1 + v3)
    

    for i in range(t1.shape[0]):
        t1[i, :] = t1[i,:] / np.linalg.norm(t1[i,:])
        t2[i, :] = t2[i,:] / np.linalg.norm(t2[i,:])

    XYZ =  np.outer(L3, v1)+ np.outer(L4, v2) + np.outer(L2,  v3)+ np.outer(L1, v4)

    shift = np.zeros_like(XYZ)
    
    for face in range(4):
        if face == 0:
            La = L1;  Lb = L2; Lc = L3;  Ld = L4        
        if face == 1:
            La = L2;  Lb = L1; Lc = L3;  Ld = L4        
        if face == 2:
            La = L3;  Lb = L1; Lc = L4;  Ld = L2        
        if face == 3:
            La = L4;  Lb = L1; Lc = L3;  Ld = L2        
            
        warp1, warp2 =WarpShiftFace3D(p, alpha, alpha, La, Lb, Lc, Ld) 
        
        blend = Lb * Lc * Ld
        denom = (Lb + 0.5 * La) *(Lc + 0.5 * La) * (Ld + 0.5 * La)
        ids = np.where(denom >  tol)[0]
        
        blend[ids] = (1 + (alpha * La[ids])**2) * blend[ids] / denom[ids]

        shift = shift + np.outer(blend * warp1, t1[face, :]) + np.outer(blend * warp2, t2[face, : ])
        lbcd = (np.array(Lb > tol, dtype = int) + np.array(Lc > tol, dtype = int) + np.array(Ld > tol, dtype = int))  < 3
        latol = La < tol

        
        
        ids = np.where(latol & lbcd)
        
        shift[ids, :] = np.outer(warp1[ids], t1[face, :]) + np.outer(warp2[ids], t2[face, :])
        
        
    XYZ = XYZ + shift

    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]
    
    return xyztorst(X, Y, Z)

def assemble_matrix_3D(ri,si,ti, monomial_basis):
    '''
    Assembling the Lagrangian Vandermonde matrix
    '''
    N = len(monomial_basis)
    Vandermonde =np.zeros((N,N))
    
    for i in range(0,N):
        for j in range(0,N):
            Vandermonde[i,j] = monomial_basis[j].subs({"r" : ri[i], "s" : si[i], "t" : ti[i]}).evalf()
    
    return Vandermonde

def generate_multivariable_monomial_basis_3D(order):
    '''
    Generates the monomial basis of a specified order in 3D
    '''
    order_list = []
    r, s, t = sympy.symbols("r s t")
    for k in range(0,order + 1):
        for l in range(0,order + 1):
            for m in range(0,order + 1):
                if(k+l+m <= order):
                    order_list.append(r**k* s**l * t**m)
    return order_list


def generate_basis_3D(ri,si,ti, N):
    '''
    Generates the Lagrangian basis of order N in 3D given node points ri, si, ti.
    '''
    monomial_basis = generate_multivariable_monomial_basis_3D(N)
    vandermonde = assemble_matrix_3D(ri, si, ti, monomial_basis)
    hij = np.linalg.inv(vandermonde)
    basis_functions =[]
    for i in range(vandermonde.shape[0]):
        basis_functions.append(sum([monomial_basis[j] * hij[j,i] for j in range(hij.shape[0])]))
    return basis_functions

def generate_Jacobi_base_3D(N):
    '''
    Generates the orthonormal basis functions in 3D of order N

    Note that the r coordinates here are different than in [1], the 0 vertex is at (0,0,0). Therefore, the normalisation is also different; 8 instead of sqrt(8).
    '''

    rm, sm, tm = sympy.symbols('r s t')

    r = 2 * rm -1
    s = 2 * sm - 1
    t = 2 * tm - 1

    bases = []
    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                if i + j + k  <= N:
                    a = - 2 * (1 + r) / ( s + t) - 1
                    b =   2 * (1 + s) / ( 1 - t) -  1
                    c = t
                    poly = 8 * jacobi_poly(i, 0, 0, a) * jacobi_poly(j, 2 * i + 1,0,b) * ( 1- b)**i * jacobi_poly(k, 2 * i + 2 * j + 2,0,c) * (1- c)**(i+j)
                    bases.append(sympy.cancel(sympy.simplify(poly)))
    return bases


# ===============================================================================================================================================================
#                                                                       Dimension Agnostic code                                                                 #
# ===============================================================================================================================================================

def Nodes(dim, N):
    if dim == 1:
        return Nodes1D(N)
    elif dim == 2:
        return Nodes2D(N)
    elif dim == 3:
        return Nodes3D(N)
    else:
        raise Exception("Dimension " + str(dim)+ " not implemented")
    
def GenerateJacobiBase(dim, N):
    if  dim == 0:
        return generate_jacobi_base_0D(N)
    elif dim == 1:
        return generate_jacobi_base_1D(N)
    elif dim == 2:
        return generate_jacobi_base_2D(N)
    elif dim == 3:
        return generate_Jacobi_base_3D(N)
    else:
        raise Exception("Dimension " + str(dim)+ " not implemented")
    
    
def GenerateLagrangianBase(dim, N, node_coords):
    if dim == 1:
        return generate_basis_1D(node_coords[0], N)
    elif dim == 2:
        return generate_basis_2D(node_coords[0], node_coords[1], N)
    elif dim == 3:
        return generate_basis_3D(node_coords[0], node_coords[1], node_coords[2], N)
    else:
        raise Exception("Dimension " + str(dim)+ " not implemented")
# ===============================================================================================================================================================
#                                                                        Code generation                                                                        #
# ===============================================================================================================================================================

def GenerateCode(self, name, order, hiname, edge_nodes):
    hii_ii_name   = hiname + "<"+str(order)+">"
    template_name = name
    N_basis_functions = self.N
    solution_vector  = "Eigen::Matrix<double, " + str(N_basis_functions)+", 1> "
    eigen_arr        = "Eigen::Matrix<double, " +str(N_basis_functions)+", " +str(N_basis_functions)+"> "
    eigen_varr       = "std::vector<Eigen::Matrix<double, " +str(N_basis_functions)+", " +str(N_basis_functions)+">> "
    cijarr           = "Eigen::Matrix<double, "+str(edge_nodes) +", " +str(edge_nodes)+"> "

    aijstart   = ("const " +eigen_arr + template_name+"::Aij  = ( eigen_arr()<< ").replace("eigen_arr", eigen_arr)
    iistart    = ("const eigen_arr " + hii_ii_name+"::Ii   = ( eigen_arr()<< ").replace("eigen_arr", solution_vector)
    cijbstart  = ("const cijarr " + template_name+"::CijB = (cijarr()<<").replace("cijarr", cijarr)
    fijstart   = ("const eigen_varr " + template_name + "::Bija = (cijarr()<<").replace("cijarr", eigen_varr)

    aijspacing = "".join(" " for i in aijstart)
    

    iispacing = "".join(" " for i in iistart)
    

    cijspacing ="".join(" " for i in cijbstart)
    cijspacing+="              "

    fijspacing ="".join(" " for i in fijstart)
    fijspacing+="              "

    def print_mat(mat, base_indent=""):
        mat = proper(mat)
        tot_string = []
        for i in range(mat.rows):
                tot_string.append(", ".join(['{:>20}'.format(str(i)) for i in mat.row(i)]))

        return (",\n" + base_indent).join(tot_string)
    
    aij = print_mat(self.Aij, base_indent=aijspacing)
    aijstring = aijstart +  aij + ").finished();"

    ii = print_mat(self.Ii, base_indent=iispacing)
    iistring = iistart +  ii + ").finished();"


    cijb= print_mat(self.baseCij, base_indent=cijspacing)
    
    cijbstring = cijbstart + cijb +").finished();"

    def get_other_array_spacing(name, mathcals):

        fijstart = ("const "+eigen_varr+" "+ template_name +"::"+name+" = {").replace("N_order", str(order))
        fijstart_spacing = "".join(" " for i in fijstart)
        eigen_array_spacing = "".join(" " for i in "(" + eigen_arr +"() << ")

        init_arrays = []
        for i in mathcals:
            imat = print_mat(i, base_indent= fijstart_spacing + eigen_array_spacing)
            init_arrays.append( "(" + eigen_arr +"() << " +  imat + ").finished()" )

        tot_init_arrays = (",\n\n" + fijstart_spacing).join(init_arrays)

        return fijstart+tot_init_arrays +"};"
    
    histring =  '''
template<>
double template_name::Hi(unsigned i, Vector local_coordinate)
{
    sb_assert(i < N_basis);
    double r = local_coordinate[0]; 
    double s = local_coordinate[1];
    double t = local_coordinate[2];
    switch(i)
    {basis_functions        
    }
    return 0.0;
}
    '''.replace("N_order", str(order)).replace("N_basis", str(N_basis_functions)).replace("template_name", hii_ii_name)
    basis_functions_epr = []

    for i, bf in enumerate(self.basis_functions):
        basis_functions_epr.append('''
        case('''+str(i)+'''):
            return ''' +
        sympy.ccode(round2(bf))+''';
        \tbreak;''')
    histring =histring.replace("basis_functions", '\n'.join(basis_functions_epr)) 
    

    base_string = ['''''']

    base_string.append(histring)
    base_string.append("template<>\n"+aijstring)
    base_string.append("template<>\n"+iistring)
    base_string.append("template<>\n"+cijbstring)
    base_string.append("template<>\n"+get_other_array_spacing("Bija",self.mathcal_Bija))

    mapping = self.Create_All_Mappings()
    def create_unsigned_maps(N_basis_functions, mapping, name, nactual = str(N_basis_functions)):
        base_mapping = ("const std::vector<Eigen::Array<unsigned, nbasis, 1>> " + template_name + "::"+name+" = {").replace("nbasis", nactual)
        base_map_spacing = "".join(" " for i in base_mapping)        
        initarrs = []
        for j, i in enumerate(mapping):            
            base =base_map_spacing +  "(Eigen::Array<unsigned, nbasis, 1>() << ".replace("nbasis", nactual)
            
            base = base + ", ".join([str(j) for j in i])+").finished()"
            initarrs.append(base)
        base_mapping= base_mapping + ",\n".join(initarrs) +"};"
        return base_mapping
    
    base_string.append("template<>\n"+create_unsigned_maps(N_basis_functions, mapping, "Vim"))
    base_string.append("template<>\n"+create_unsigned_maps(len(self.surface_maps[0]), self.surface_maps, "Sv", str(edge_nodes)))
    
    
    return '\n\n'.join(base_string)

def EdgeFunction2D(node_coords, i):    
    if(len(node_coords[0]) == 1):
        return [0]
    else:
        return np.where([np.abs(node_coords[0] + node_coords[1]) <=1e-4,  np.abs(node_coords[0] + 1) <= 1e-4, np.abs(node_coords[1] + 1) <= 1e-4][i])[0]

def EdgeFunction3D(node_coords, i):
    r = node_coords[0]
    s = node_coords[1] 
    t = node_coords[2]
    if(len(r) == 1):
        return [0]
    else:
        return np.where([np.abs(1 - r - s  - t) < 1e-8, np.abs(r) < 1e-8, np.abs(s) < 1e-8, np.abs(t) < 1e-8][i])[0]

def EdgeFunction1D(node_coords, i):
    
    if len(node_coords[0]) == 1:
        return [0]
    else:
        return np.where([np.abs(node_coords[0] - 1)<= 1e-8, np.abs(node_coords[0] + 1)<= 1e-8][i])[0]
    
class DGM_Neutronics_Elements:
    def __init__(self, order, basis_functions, jacobi_basis_functions,  symbols, surface_integration,
                                                                     node_coords,
                                                                     number_vertices,
                                                                     edge_function, vertices_coords):
        self.order           = order
        self.N               = len(basis_functions)
        self.jacobi_funcs    = jacobi_basis_functions
        self.Nr              = range(len(basis_functions))

        self.basis_functions    = basis_functions
        self.symbols            = symbols
        self.dim                = len(self.symbols)                
        
        self.node_coords        = node_coords
        self.surface_maps       = [edge_function(node_coords, i ) for i in range(number_vertices)]        
        self.nonzero_surf       = len(self.surface_maps[0])
        self.vertices_coords = vertices_coords        
        self.surface_im = [tuple(np.roll(np.arange(0,number_vertices), -i )) for i in range(len(self.surface_maps))]


        
        self.node_coords_rep = [{self.symbols[k] : self.node_coords[k][i] for k in range(len(self.symbols))} for i in self.Nr]

        self.vandermonde = CreateVanderMonde(self.node_coords_rep, self.jacobi_funcs)
                                         
    
    def Calculate_Ii(self):        
        factor =  float(self.jacobi_funcs[0].evalf() )
        self.Ii = sympy.Matrix((np.linalg.inv(self.vandermonde.T * factor ))[:,0] )

    def Calculate_Aij(self):
        self.Aij = sympy.Matrix(np.linalg.inv(self.vandermonde@ self.vandermonde.T))

    def Calculate_Bij(self):
        dvdm_a = [np.zeros((self.N, self.N)) for a in range(self.dim)]
        self.mathcal_Bija = [np.zeros((self.N, self.N)) for a in range(self.dim)]
        
        for a in range(self.dim):
            diff_jacobi = [sympy.diff(self.jacobi_funcs[k],self.symbols[a]) for k in self.Nr]
            for i in self.Nr:
                for j in self.Nr:
                    dvdm_a[a][i,j] = diff_jacobi[j].subs(self.node_coords_rep[i])
            self.mathcal_Bija[a] = - (self.Aij @ ( dvdm_a[a] @  np.linalg.inv(self.vandermonde) )).T        
    
    def Calculate_Cij(self):                
        
        if self.dim == 3:
            nodes_surface_0_coords = [2 * self.node_coords[i][self.surface_maps[0]]  - 1 for i in range(self.dim - 1)]
        else:
            nodes_surface_0_coords = [self.node_coords[i][self.surface_maps[0]] for i in range(self.dim - 1)]
        jacobi_base_surf = GenerateJacobiBase(self.dim - 1, self.order)
        node_coords_replace_surface_0 = [{self.symbols[k] : nodes_surface_0_coords[k][i] for k in range(self.dim - 1)} for i in range(len(self.surface_maps[0]))]        
        surf_vandermonde = CreateVanderMonde(node_coords_replace_surface_0, jacobi_base_surf)
        self.baseCij = sympy.Matrix(np.linalg.inv(surf_vandermonde @ surf_vandermonde.T))
        if self.dim == 3:
            self.baseCij = proper(self.baseCij  / 4)
        else:
            self.baseCij = proper(self.baseCij)
        

    def Create_Mapping(self, im):
        qib = [bar(i, self.symbols) for i in self.vertices_coords]
        qi  = self.vertices_coords
        
        
        irep = [np.argwhere(np.array(im) == i)[0][0] for i in range(1,len(qi))]
        
        polys = [i.subs({qib[j]  : qi[irep[j- 1]] for j in range(1, len(qi))}) for i in qib[1:]]

        fpolys = [sympy.lambdify(self.symbols, i, "numpy") for i in polys]

        def almost_same(a,b, eps = 1e-5):
            return np.abs(a - b) < eps
        
        nodecoords_new = [i(*self.node_coords) for i in fpolys]
        
        permutation = []
        for j in range(self.node_coords[0].shape[0]):
            for i in range(self.node_coords[0].shape[0]):
                if all([almost_same(self.node_coords[k][i], nodecoords_new[k][j]) for k in range(len(nodecoords_new))]):
                    permutation.append(i)

        return permutation
    
    def Create_All_Mappings(self):
        vertices_list = list(np.arange(len(self.vertices_coords)))
        all_index_maps = list(itertools.permutations(vertices_list))
        mapping_list = []
        for im in all_index_maps:
            mapping_list.append(self.Create_Mapping(im))

        return mapping_list
    
    def Calculate_All(self):
        self.Calculate_Aij()
        self.Calculate_Bij()
        self.Calculate_Cij()
        self.Calculate_Ii()


class Line1D(DGM_Neutronics_Elements):
    def __init__(self, N):
        ri    = Nodes(1, N)
        basis = GenerateLagrangianBase(1, N, [ri])
        jacobi = GenerateJacobiBase(1, N)
        r     = sympy.symbols('r')        
        vert_coords = [-r , r] # in fact (1-r)/2 and (1+r)/2 but has been transferred to a form with r only on  the right side.        
        DGM_Neutronics_Elements.__init__(self,N, basis, jacobi,(r,), None, (ri, ), 2,  EdgeFunction1D, vert_coords)

class Triangle2D(DGM_Neutronics_Elements):
    def __init__(self, N):
        ri, si    = Nodes(2, N)
        basis = GenerateLagrangianBase(2, N, [ri, si])
        jacobi = GenerateJacobiBase(2, N)
        r,s  = sympy.symbols('r s')
        vert_coords = [- r -s - 1, r, s] # these are in fact (-r - s) / 2, (r + 1 ) / 2, (s+1) / 2 but have been transferred to forms with r, s to allow substitution!        
        DGM_Neutronics_Elements.__init__(self,N, basis, jacobi, (r,s), None, (ri,si), 3,  EdgeFunction2D, vert_coords)


class Tetrahedron3D(DGM_Neutronics_Elements):
    def __init__(self, N):
        ri, si, ti    = Nodes(3, N)
        basis = GenerateLagrangianBase(3, N, [ri, si, ti])
        jacobi = GenerateJacobiBase(3, N)

        r,s, t  = sympy.symbols('r s t')
        vert_coords = [1 - r -s -t, r, s, t] # these are not the same as in Hesthaven, but a standard tetrahedron with (0,0) at vertex 0
        
        DGM_Neutronics_Elements.__init__(self, N, basis, jacobi, (r,s,t), None, (ri, si, ti), 4,  EdgeFunction3D, vert_coords)

def convert_old_coordinates(element, coords_old):
    if element == "interval":
        return (coords_old + 1) / 2.0
    elif element == "triangle":
        return (coords_old + 1) / 2.0
    elif element == "tetrahedron":
        return coords_old
    else:
        raise ValueError("Element type " + str(element) + " not recognized")



def create_permutation(element, element_old, element_new):
    import numpy as onp
    old_coords = onp.atleast_2d(onp.array(convert_old_coordinates(element, onp.array(element_old.node_coords)))).T
    new_coords = onp.array(element_new.element.points)    
    idx_list = []   
    for i in range(new_coords.shape[0]):        
        idx = onp.where(onp.all(onp.isclose(old_coords, new_coords[i:i+1, :]), axis=1))
        if idx[0].shape[0] != 1:
            raise ValueError("Could not find matching coordinate for new coord:", new_coords[i, :], " in new coords")
        idx_list.append(idx[0][0])
    return onp.array(idx_list)


def get_factor_difference(element):
    if element == "interval":
        return 1/2.0
    elif element == "triangle":
        return 1/4.0
    elif element == "tetrahedron":
        return 1.0
    else:
        raise ValueError("Element type " + str(element) + " not recognized")
    


def get_jacobian_difference(element):
    if element == "interval":
        return 2.0
    elif element == "triangle":
        return 2.0
    elif element == "tetrahedron":
        return 1.0
    else:
        raise ValueError("Element type " + str(element) + " not recognized")