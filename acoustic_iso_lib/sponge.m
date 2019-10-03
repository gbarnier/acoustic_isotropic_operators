U_0=0.001
alpha_1=0.06
alpha_2=0.18
alpha_3=0.4

x = 0:30

y1 = cosh(alpha_1.*x)
y1 = U_0./y1./y1
y2 = cosh(alpha_2.*x)
y2 = U_0./y2./y2
y3 = cosh(alpha_3.*x)
y3 = U_0./y3./y3
figure;
hold;
plot(x,y1);
plot(x,y2);
plot(x,y3);
legend;
