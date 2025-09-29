% test_rqpso_F1_getF.m
clc; clear; close all;

[LB, UB, Dim, F_obj] = Get_F('F1');   % requires Get_F.m on path

PopSize = 10;
MaxIter = 300;

[BestSol, BestFitness, Curve, ~] = RQPSO1(PopSize, MaxIter, UB, LB, Dim, F_obj);

fprintf('BestFitness = %.6e\n', BestFitness);

figure('Color','w');
plot(Curve, 'LineWidth', 2); grid on;
xlabel('Iteration'); ylabel('Best-so-far'); title('RQPSO1 on F1');
