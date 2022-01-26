clc; clear; close all;

pod = readtable('CTG.csv');
rng(50)

%% Ucitavanje podactaka
%ulaz = [pod.b, pod.e, pod.LBE, pod.LB, pod.AC, pod.FM, pod.UC, pod.ASTV, pod.MSTV, pod.ALTV, pod.MLTV, pod.DL, pod.DS, pod.DP, pod.DR, pod.Width, pod.Min, pod.Max, pod.Nmax, pod.Nzeros, pod.Mode, pod.Mean, pod.Median, pod.Variance, pod.Tendency];
%izlaz = [pod.A, pod.B, pod.C, pod.D, pod.E, pod.AD, pod.DE, pod.LD, pod.FS, pod.SUSP, pod.CLASS, pod.NSP];

ulaz = [pod.b, pod.e, pod.LBE, pod.LB, pod.AC, pod.FM, pod.UC, pod.ASTV, pod.MSTV, pod.ALTV, pod.MLTV, pod.DL, pod.DS, pod.DP, pod.DR, pod.Width, pod.Min, pod.Max, pod.Nmax, pod.Nzeros, pod.Mode, pod.Mean, pod.Median, pod.Variance, pod.Tendency, pod.A, pod.B, pod.C, pod.D, pod.E, pod.AD, pod.DE, pod.LD, pod.FS, pod.SUSP];
izlaz = pod.NSP;

ulaz = ulaz';
izlaz = izlaz';

%% Prikaz raspdele odbiraka po klasama
figure
histogram(izlaz)

%% razdvajanje po klasama

k1 = ulaz(:,izlaz==1);
k2 = ulaz(:,izlaz==2);
k3 = ulaz(:,izlaz==3);

%% Podela podataka

rng(50);
N1 = length(k1);
ind = randperm(N1);

k1tr = k1(:,ind(1:0.6*N1));
k1te = k1(:,ind(0.6*N1+1:0.8*N1));
k1val = k1(:,ind(0.8*N1+1:N1));

rng(50);
N2 = length(k2);
ind = randperm(N2);

k2tr = k2(:,ind(1:0.6*N2));
k2te = k2(:,ind(0.6*N2+1:0.8*N2));
k2val = k2(:,ind(0.8*N2+1:N2));

rng(50);
N3 = length(k3);
ind = randperm(N3);

k3tr = k3(:,ind(1:ceil(0.6*N3)));
k3te = k3(:,ind(ceil(0.6*N3+1):ceil(0.8*N3)));
k3val = k3(:,ind(ceil(0.8*N3+1):N3));

ulazTr = [k1tr, k2tr, k3tr];
g1 = length(k1tr);
g2 = g1 + length(k2tr);
g3 = g2 + length(k3tr); 
izlazTr = zeros(3,length(k1tr)+length(k2tr)+length(k3tr));
izlazTr(1,1:g1)=1;
izlazTr(2,g1+1:g2)=1;
izlazTr(3,g2+1:g3)=1;

ulazVal = [k1val, k2val, k3val];
g1 = length(k1val);
g2 = g1 + length(k2val);
g3 = g2 + length(k3val); 
izlazVal = zeros(3,length(k1val)+length(k2val)+length(k3val));
izlazVal(1,1:g1)=1;
izlazVal(2,g1+1:g2)=1;
izlazVal(3,g2+1:g3)=1;

ulazTe = [k1te, k2te, k3te];
g1 = length(k1te);
g2 = g1 + length(k2te);
g3 = g2 + length(k3te); 
izlazTe = zeros(3,length(k1te)+length(k2te)+length(k3te));
izlazTe(1,1:g1)=1;
izlazTe(2,g1+1:g2)=1;
izlazTe(3,g2+1:g3)=1;

%% mesanje unutar skupa
N = length(ulazTr);
ind = randperm(N);
ulazTr = ulazTr(:,ind);
izlazTr = izlazTr(:,ind);

N = length(ulazVal);
ind = randperm(N);
ulazVal = ulazVal(:,ind);
izlazVal = izlazVal(:,ind);

N = length(ulazTe);
ind = randperm(N);
ulazTe = ulazTe(:,ind);
izlazTe = izlazTe(:,ind);

% spajanje trening i validacioni
ulazSve = [ulazTr, ulazVal];
izlazSve = [izlazTr, izlazVal];

%% Formiranje mreze

arhitektura = {[10, 5], [12, 6, 3], [4 5 6]};
Abest = 0;
F1best = 0;

for reg = [0.01, 0.1, 0.3]
    for w = [1, 1.2, 1.5, 2]
        for lr = [1, 0.5, 0.05, 0.005]
            for arh = 1:length(arhitektura)
                rng(5)
                net = patternnet(arhitektura{arh});
                
                for i = 1:length(arhitektura{arh})
                    net.layers{i}.transferFcn = 'tansig';
                end
                
                net.divideFcn = 'divideind';
                net.divideParam.trainInd = 1 : length(ulazTr);
                net.divideParam.valInd = length(ulazTr)+1 : length(ulazSve);
                net.divideParam.testInd = [];

                net.performParam.regularization = reg;

                net.trainFcn = 'traingdm';

                net.trainParam.lr = lr;
                net.trainParam.epochs = 2500;
                net.trainParam.goal = 1e-4;
                net.trainParam.max_fail = 20;

                weight = ones(1, length(izlazSve));
                weight(:,izlazSve(1,:) == 1) = w;

                [net, info] = train(net, ulazSve, izlazSve, [], [], weight);

                pred = sim(net, ulazVal);
                [v, ind] = max(pred);
                out = zeros(3,length(pred));
                out(1,ind==1)=1;
                out(2,ind==2)=1;
                out(3,ind==3)=1;

                [~, cm] = confusion(izlazVal, out);
                A = 100*sum(trace(cm))/sum(sum(cm));
                cm=cm';
                prec = zeros(3,1);
                rec = zeros(3,1);
                fi = zeros(3,1);
                for i = 1:3
                    prec(i,1) = cm(i,i)/(cm(i,1)+cm(i,2)+cm(i,3));
                    rec(i,1) = cm(i,i)/(cm(1,i)+cm(2,i)+cm(3,i));
                    fi(i,1) = 2*prec(i,1)*rec(i,1)/(prec(i,1)+rec(i,1));
                end
                F1 = sum(fi(:,1))/3*100;

                disp(['Reg = ' num2str(reg) ', ACC = ' num2str(A) ', F1 = ' num2str(F1)])
                disp(['LR = ' num2str(lr) ', epoch = ' num2str(info.best_epoch) ',weight = ' num2str(w) 'reg = ' num2str(reg)])

                if F1 > F1best
                    F1best = F1;
                    Abest = A;
                    reg_best = reg;
                    w_best = w;
                    lr_best = lr;
                    arh_best = arhitektura{arh};
                    ep_best = info.best_epoch;
                end
            end
        end
    end
end

%% Treniranje NM sa optimalnim parametrima (na celom trening + val skupu)
net = patternnet(arh_best);

net.divideFcn = '';

for i = 1:length(arh_best)
    net.layers{i}.transferFcn = 'tansig';
end
                
net.performParam.regularization = reg_best;

net.trainFcn = 'traingdm';

net.trainParam.lr = lr_best;

net.trainParam.epochs = ep_best;
net.trainParam.goal = 1e-4;

weight = ones(1, length(izlazSve));
weight(:,izlazSve(1,:) == 1) = w_best;

[net, info] = train(net, ulazSve, izlazSve, [], [], weight);

%% Performanse NM
pred = sim(net, ulazTe);
[v, ind] = max(pred);
out = zeros(3,length(pred));
out(1,ind==1)=1;
out(2,ind==2)=1;
out(3,ind==3)=1;
figure, plotconfusion(izlazTe, out);