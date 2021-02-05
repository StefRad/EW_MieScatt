%codice
code = 'MLCT-OW, A';
%forma
shape = 'triangolare'; %'rettangolare'

%k=0.03;
k=0.07;
%material density (kg/m^3)
%rho = 3.17*1e3;
rho=2.37*1e3;
% thickness(m)
%t = 5e-7;
t=0.515*1e-6;
% width(m)
%w = 1e-5;
w=22e-6;
if string(shape) == 'triangolare'
    disp('triangolo');
    w=2*w;
end
% resonance frequency(Hz)
f0 = 73e3;
%omega0 = 2*pi*1e5;
omega0 = 2*pi*f0;
%dynamic viscosity (kg / m*s) at 300K
mu = 1.872*1e-5; 
% thermal energy
RT = 8.31*300;
E = RT/(14*1e-3); %(energy) / (molecular weight air=28.96*1e-3)

%pression (Pa) for values under 100 Pa the air is not a fluid for the
%calculation
p = 1:100000;

[q,n,d] = q_factor(rho,t,w,omega0,mu,E,p);
y1 = q_factor_mol(rho,t,w,omega0,mu,E,p);
n=n(1);
d=d(1);

size(q)
size(y1)
h(1) = figure;
loglog(p,q,'o-');
hold on;
loglog(p,y1,'o-');
%hold on;
%loglog(p,q.*y1./(q+y1),'o-');
grid on;

ylabel('Q-value');
xlabel('pressione [Pa]');
c = newline;
annotation('textbox', 'string', 'cantilever: ' + string(code) + c +...
    'k = ' + string(k) + ' N/m '+ c + 'f_0 = '+ ... 
    string(f0/1000)+ ' kHz' + c + ' \rho = ' + string(rho) + ' kg/m^3' + c + ...
    'spessore = ' + string(t*1000000) + ' \mum' + c + ...
    'larghezza = ' + string(w*1000000) + ' \mum' + c + ...
    'forma: ' + string(shape) , 'FitBoxToText','on')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%m=1.6e-11;
Q=max(q,y1);

F=1e-16;
F2=2e-16;

m = k / (f0*6.28)^2;

%f = 1:0.1:20000;

f1 = f0 * sqrt(1 - (1 ./ (2*Q.^2)) );

%calcolo rumore termico
z = 0.64e-10/sqrt(k);

%ampiezza alla nuova frequenza di risonanza
a1 = ampiezza(6.28*f0, F, m, Q, 6.28*f1);
a4 = ampiezza(6.28*f0, F2, m, Q, 6.28*f1);

h(2) = figure;
loglog(p,a1,p,a4);
yline(z);

ylabel('ampiezza di oscillazione in risonanza [m]');
xlabel('pressione [Pa]');

str1 = 'Forza applicata: '+string(F*10^12)+' pN';
str2 = 'Forza applicata: '+string(F2*10^12)+' pN';
str3 = 'Deviazione standard rumore termico';
legend(str1,str2);


annotation('textbox', 'string', 'cantilever: ' + string(code) + c +...
    'k = ' + string(k) + ' N/m '+ c + 'f_0 = '+ ... 
    string(f0/1000)+ ' kHz' + c + ' \rho = ' + string(rho) + ' kg/m^3' + c + ...
    'spessore = ' + string(t*1000000) + ' \mum' + c + ...
    'larghezza = ' + string(w*1000000) + ' \mum' + c + ...
    'forma: ' + string(shape), 'FitBoxToText','on')

path = cd;
build_path = strcat(path,'\pression_Q-value_amplitude\', string(code),'.fig');
%savefig(h,build_path);

function y = ampiezza(omega0, forza, massa, quality, omega)
n = forza/massa;
d = sqrt(((omega0^2 - omega.^2 ).^2) + (((omega0*omega) ./ quality).^2) );
y = n ./ d;
end


function [y,n,d] = q_factor(r,t,w,w0,m,E,p)
n = 2*r*t*(w^2)*w0;
d = (6*pi*m*w) + ((3/2)*pi*(w^2)*sqrt((2*m*w0*p)/(E)));
y = n * (d.^(-1));
end

function y1 = q_factor_mol(r,t,w,w0,m,E,p)
y1 = (t*r*w0/4)*sqrt(pi/2)*sqrt(E)*1./p;
end

