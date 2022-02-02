import matplotlib.pyplot as py

''' Gamma Correction: V_out = A * (V_in)^(gamma)
    Usually, A = 1 and in this case too
'''

x_axis = [i for i in range(256)]
A = [1 for i in range(256)]

#### gamma_0_25 = 0.25 ####
gamma_0_25 = 0.25

V_in = [i**gamma_0_25 for i in range(256)]
# print(V_in)
# print(V_out)
V_out = [a*b for a,b in zip(V_in, A)]
# print(V_out)
V_out = [i*255/4 for i in V_out]                    #scaled to range between 0-255
py.plot(x_axis, V_out, label = 'gamma = 0.25')

#### gamma_0_50 = 0.50 ####
gamma_0_50 = 0.50

V_in = [i**gamma_0_50 for i in range(256)]
V_out = [a*b for a,b in zip(V_in, A)]
V_out = [i*255/16 for i in V_out]                   #scaled to range between 0-255
py.plot(x_axis, V_out, label = 'gamma = 0.50')

#### gamma_1_00 = 1 ####
gamma_1_00 = 1

V_in = [i**gamma_1_00 for i in range(256)]
V_out = [a*b for a,b in zip(V_in, A)]               #scaled to range between 0-255
py.plot(x_axis, V_out, label = 'gamma = 1.00')

#### gamma_1_50 = 1.50 ####
gamma_1_50 = 1.50

V_in = [i**gamma_1_50 for i in range(256)]
V_out = [a*b for a,b in zip(V_in, A)]
V_out = [i*255/4075 for i in V_out]                 #scaled to range between 0-255
py.plot(x_axis, V_out, label = 'gamma = 1.50')

#### gamma_2_00 = 2 ####
gamma_2_00 = 2

V_in = [i**gamma_2_00 for i in range(256)]
V_out = [a*b for a,b in zip(V_in, A)]
V_out = [i*255/65025 for i in V_out]                #scaled to range between 0-255
py.plot(x_axis, V_out, label = 'gamma = 2.00')


py.xlabel('x - axis: Input Value')
py.ylabel('y - axis: Output Value')
py.title('Gamma Correction for better human eye perception (Scaled)')
py.legend()
py.show()
