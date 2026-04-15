import re #expresiones regulares

'''
=================================================================
Funciones Utiles
===============================================================
'''

def validar_operador(op):
    # Patrón: +_ o -_ seguido de un número entero (uno o más dígitos)
    patron = r'^[\+\-]_\d+$'
    return bool(re.match(patron, op))


def multiplicar_pauli(a, b):
    """Multiplica dos matrices de Pauli representadas como 'I','X','Y','Z'"""
    if a == 'I':
        return 1, b
    if b == 'I':
        return 1, a
    if a == b:
        return 1, 'I'
    
    # reglas básicas
    reglas = {
        ('X','Y'):(1j,'Z'),
        ('Y','X'):(-1j,'Z'),
        ('Y','Z'):(1j,'X'),
        ('Z','Y'):(-1j,'X'),
        ('Z','X'):(1j,'Y'),
        ('X','Z'):(-1j,'Y')
    }
    return reglas[(a,b)]

def multiplicar_tuplas(op1, op2):
    coef1, lista1 = op1
    coef2, lista2 = op2
    coef_total = coef1 * coef2
    lista_result = []
    
    for a,b in zip(lista1, lista2):
        c, p = multiplicar_pauli(a,b)
        coef_total *= c
        lista_result.append(p)
    
    return coef_total, lista_result  

def unir_diccionarios(lista_dic):
    resultado = {}

    for dic in lista_dic:
        for clave, valor in dic.items():
            if clave in resultado:
                resultado[clave] += valor
            else:
                resultado[clave] = valor

    # quitar términos numéricamente cero
    resultado_final = {}

    for clave, valor in resultado.items():
        if abs(valor) > 1e-12:
            resultado_final[clave] = valor

    return resultado_final

'''
=================================================================
IMPLEMENTACIÓN : TRANSFORMACIÓN DE JORDAN WIGNER
===============================================================
'''

def JWapp(chain: str,n):
    '''
    Verificación de buen funcionamiento. 
    Se esperan cadenas del tipo : '+_0 -_1' , con espacios entre ellas. 
    '''
    ops = chain.split()

    # Verificar cada operador
    for op in ops:
        if validar_operador(op):
            continue 
        else:
            raise ValueError(f"{op} → formato INCORRECTO")
        
    #Aplicación de Jordan-Wigner:

    JW=[]
    for op in ops:
        U=[]
        if op:
            if op[0]=="+":
                p=1
            else:
                p=0
        s=int(op[2:])
        V=[]
        for i in range(s):
            V.append("Z") 
        V.append("X") 
        for i in range(s+1,n):
            V.append("I") 
        U.append((0.5,V)) 
        W=[]
        for i in range(s):
            W.append("Z") 
        W.append("Y") 
        for i in range(s+1,n):
            W.append("I") 
        U.append((0.5j*(-1)**p,W)) 
        JW.append(U)

    # Multiplicamos terminos: 

    R=[]
    for i in [0,1]:
        for j in [0,1]:
            R.append(multiplicar_tuplas(JW[0][i], JW[1][j]))
    for k in range(2,len(JW)):
        T=[]
        for i in range(len(R)):
         for j in [0,1]:
            T.append(multiplicar_tuplas(R[i], JW[k][j])) 
        R=T  


    #Simplificamos  y  convertimos a diccionario:

    acumulado={}
    for coef, PauliOp in R:
        key = tuple(PauliOp)

        if key in acumulado:
            acumulado[key] += coef
        else:
            acumulado[key] = coef
    R_dict = {}

    for clave, coef in acumulado.items():
        if abs(coef) > 1e-12:
            R_dict[clave] = coef
    for clave in R_dict:
        if len(clave) != n:
            raise ValueError(f"Salió una clave de longitud {len(clave)} en vez de {n}: {clave}")
      
    return R_dict


def JWdic(data: dict,n):
    S=[]
    for ferm_op_EX, coef_EX in data.items():
        f=JWapp(ferm_op_EX,n) 
        f_escalado={}
        for ferm_op_IN, coef_IN in f.items():
            f_escalado[ferm_op_IN]=coef_IN*coef_EX 
        S.append(f_escalado) 
        H = unir_diccionarios(S)
        for clave in H:
            if len(clave) != n:
                raise ValueError(f"Después de unir, longitud incorrecta: {clave}") 
    return H
    




