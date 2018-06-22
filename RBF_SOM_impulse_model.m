%% Using RBF and Kohonen's Neural Networks to model impulse response of dynamic system
% Kohonen's network determine location of RBF centers
% Comments in Polish
% Author: Pawe³ Mêdyk

% Wykorzystanie sieci neuronowych RBF i Kohonena do modelowania odpowiedzi
% impulsowej uk³adu dynamicznego
% Sieæ Kohonena wyznacza centra sieci RBF. Sieæ RBF modeluje odpowiedŸ
% impulsow¹.
% Imiê i nazwisko autora: Pawe³ Mêdyk

% import danych wejsciowych
close all;
baza_ucz_we=importdata('dane/zad2_dane_we.txt');
baza_ucz_wy=importdata('dane/zad2_dane_wy.txt');

% stworz macierz wejsc i zapamietaj jej rozmiary
wej=[baza_ucz_we; baza_ucz_wy];
N=size(wej,1);
P=size(wej,2);

mse_ucz=1;          % inicjacja bledu sredniokwadratowego miedzy odp. sieci a oczekiwanym wyjsciem 
licznik_iteracji=0; % inicjacja liczby iteracji; dla informacji
while mse_ucz > 0.001 % warunek wyjscia - MSE <  0.001
    licznik_iteracji=licznik_iteracji+1; % rozpocznij kolejna iteracje
    K=50;           % liczba neuronow w sieci
    kol=sqrt(K);    % liczba kolumn
    C=zeros(K,2);
    for k=1:K
        C(k,:)=[floor((k-1)/kol)+1, mod(k-1,kol)+1]; % nadanie wspolrzednych na mapie
    end  % rozmieszczenie neuronow w przestrzeni
    % siec o strukturze 2D - kwadratowa (prostokat o rownych
    % odleglosciach miedzy punktami) dist(n(i),n(i+1))=1 dla kazdego
    % neuronu; jeden wiersz zawiera liczbê neuronów równ¹ 'kol'
    % n1       n2          n3          n4 ...
    % n(kol+1) n(kol+2)    n(kol+3)    n(kol+4) ...

    %struktura sieci i wagi
    W=zeros(N,K);               % macierz wag
    a=-1; b=15;                 % przedzial inicjowanych wag (przestrzen x)
    W(1,:)=(b-a)*rand(1,K)+a;   % inicjuj wektor wag
    a=0; b=2;                   % przedzial inicjowanych wag (przestrzen y)
    W(2,:)=(b-a)*rand(1,K)+a;   % inicjuj wektor wag
    WLOS=W;                     % zapamietaj macierz wag do wykresu
    dist = @(v1,v2) sqrt(sum((v2-v1).^2)); % odleglosc - norma euklidesowa
    %neighbour_f= @(d,lam) (d<lam).*1;          % f. sasiedztwa prostokatna
    neighbour_f= @(d,lam) exp(-d^2/(2*lam^2)); % f. sasiedztwa Gaussa
    Epoki=2000; % liczba epok
    w=1/Epoki;  % czestotliwosc zmian parametrow na epoke
    alpha=1.3;  % wspolczynnik uczenia sie
    lambda=3.5; % promien sasiedztwa
    D=zeros(K,1);   % wektor odleglosci w przestrzeni danych wej.
    for ep=1:Epoki
        L=randi([1 P],1);   % wylosuj przyklad ucz.
        for k=1:K;          % odl neuronow od wejsc
            D(k)= dist(wej(:,L),W(:,k)); % D(k) odleglosc k'tego neuronu od wyl.danych
        end
        [winn,w_i]=min(D); % wyznacz zwycieski neuron
        for k=1:K;
            map_k_dist=dist(C(w_i,:),C(k,:));   % odleglosc k'tego neuronu na mapie od zwyciezcy
            W(:,k)= W(:,k)+alpha*neighbour_f(map_k_dist,lambda)*(wej(:,L)-W(:,k)); % m.subtraktywna 
        end
        %redukcja parametrow
        alpha=(1-w)*alpha;
        lambda=(1-w)*lambda;
    end % zakonczenie 1 cyklu
    
    %% siec RBF
    x=wej(1,:)';    % wektor wejsc
    d=wej(2,:)';	% oczekiwany wektor wyjsc
    c=W(1,:)';      % centra funkcji radialnej
    p=length(x);	% rozmiar wektora wejsc

    % wyznaczanie wspolczynnika ts, do parametru szerokosci
    ts=(c(1,:)-c(2,:))'*(c(1,:)-c(2,:));
    for i=1:K
        for j=i:K
            ts=max(ts,(c(i,:)-c(j,:))'*(c(i,:)-c(j,:)));
        end
    end
    sigma = ts/K;      % parametr szerokosci radialnej funkcji bazowej
    
    % radialna funkcja bazowa - gaussa |   -r^2/(2*sigma^2)
    phi=@(x,c) exp(-((x-c)'*(x-c))/(2*sigma^2));
    
    % wypelnienie macierzy pobudzeñ Phi
    Phi = zeros(p,K+1);
    Phi(:,1)=1;	% wejscie progowe
    for i=1:p
        for j=1:K
            Phi(i,j+1)=phi(x(i),c(j));
        end
    end
    Phi_odwr=(Phi'*Phi)\(Phi'); % macierz pseudoodwrotna macierzy Phi
    w=Phi_odwr*d;               % wyznacz wektor wag 
    
   
    d_siec = Phi*w;         % odp sieci na dane ucz.
    mse_ucz=immse(d_siec,d) % MSE odp. sieci wzgledem wyjsc oczekiwanych
    
end
    
%% Odpowiedz sieci na zestaw testowy
x_test=[0:0.01:x(end)]';    % wczytaj zestaw testowy
p_test=length(x_test);

Phi = zeros(p_test,K+1);    % macierz pobudzen neuronow wartswy ukrytej
Phi(:,1)=1;         % wejscie progowe
for i=1:p_test
    for j=1:K
        Phi(i,j+1)=phi(x_test(i),c(j));
    end
end

d_siec_test = Phi*w;    % odp. sieci na dane testowe

% plot sieci Kohonena
figure(1); hold on;
plot(wej(1,:),wej(2,:), '*g');
plot(W(1,:),W(2,:),'or')
plot(WLOS(1,:),WLOS(2,:),'+b')
legend('dane uczace','odp.sieci')
title('Odwzorowanie klastrow. Siec Kohonena')

% plot danych uczacych, reakcji sieci RBF na dane uczace i polozenia centrow
figure(2); hold on;
plot(x,d_siec,'-k');
plot(x,d,'*r')
plot(c,W(2,:),'sk')
str_title=sprintf('Identyfikacja czlonu dynamiki.\nOdp. sieci na zestaw uczacy wzgledem oryginalu.');
title(str_title);
legend('odp.sieci na d.uczace','dane uczace','centra RBF');

%% plot danych uczacych, reakcji sieci RBF na dane testowe i polozenia centrow
figure(3); hold on;
plot(x_test,d_siec_test,'-b')
plot(x,d,'.r')
plot(c,W(2,:),'sk')
str_title=sprintf('Identyfikacja czlonu dynamiki.\nOdp. sieci na zestaw testowy wzgledem oryginalu.');
title(str_title);
legend('odp. na zest. test.','dane uczace','centra RBF');
licznik_iteracji