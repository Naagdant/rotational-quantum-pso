function [BestSol, BestFitness, Curve, PopHistRun] = RQPSO1(PopSize, MaxIter, UB, LB, Dim, F_obj)
    % RQPSO: Quantum Rotational PSO based on multi-qubit probability amplitudes
    % Inspired by quantum rotation gates and multi-qubit encoding for particles
    % Improved: increased n_qubits for better representation, tuned theta0_delta to linear decrease for balanced exploration, higher c1/c2 for stronger attraction, adjusted mutation and stagnation for low-iteration performance, added final local search refinement
    N = PopSize;
    % Handle scalar bounds
    if numel(LB) == 1
        LB = LB * ones(1, Dim);
        UB = UB * ones(1, Dim);
    end
    range = UB - LB;
    
    % Number of qubits with extra for better representation
    n_qubits = ceil(log2(Dim)) + 5;  % Increased to +5 for enhanced flexibility and better coverage in high dimensions
    
    % Initialize phases theta in [0, 2*pi)
    theta = 2 * pi * rand(N, n_qubits);
    
    % Compute initial positions using multi-qubit amplitudes
    X = zeros(N, Dim);
    for i = 1:N
        psi = compute_amplitudes(theta(i, :));
        psi = psi(1:Dim);
        X(i, :) = LB + range .* (psi' + 1) / 2;
    end
    
    % Evaluate initial fitness
    Fitness = zeros(N, 1);
    for i = 1:N
        Fitness(i) = F_obj(X(i, :));
    end
    
    % Initialize personal and global bests
    pbest_theta = theta;
    pbest_fitness = Fitness;
    [gbest_fitness, gbest_idx] = min(pbest_fitness);
    gbest_theta = pbest_theta(gbest_idx, :);
    gbest = X(gbest_idx, :);
    
    % Parameters
    %c1 = 1.5; c2 = 1.5; 
   c1 = 2.05; c2 = 2.05;  % Slightly higher coefficients for stronger attraction
    mutation_prob = 0.15;  % Balanced for diversity
    stagnation_limit = 3;  % Balanced for quick response in low iterations
    stagnation_count = 0;
    
    % Storage
    Curve = zeros(MaxIter, 1);
    PopHistRun = cell(MaxIter, 1);
    
    for it = 1:MaxIter
        % Decreasing rotation step size (linear decrease for better balance)
        theta0_delta = pi * (1 - it / MaxIter);
        
        old_gbest_fitness = gbest_fitness;
        
        % Update phases for all particles
        for i = 1:N
            for j = 1:n_qubits
                theta_ij = theta(i, j);
                theta_bij = pbest_theta(i, j);
                theta_gj = gbest_theta(j);
                
                % Delta for personal best (shorter path)
                diff_b = mod(theta_bij - theta_ij + pi, 2*pi) - pi;
                delta_theta_bij = c1 * rand * diff_b / pi * theta0_delta;  % Normalized diff for controlled step
                
                % Delta for global best (shorter path)
                diff_g = mod(theta_gj - theta_ij + pi, 2*pi) - pi;
                delta_theta_gj = c2 * rand * diff_g / pi * theta0_delta;
                
                delta_theta_ij = delta_theta_bij + delta_theta_gj;
                
                theta(i, j) = theta(i, j) + delta_theta_ij;
            end
        end
        
        % Quantum NOT-like mutation
        for i = 1:N
            if rand < mutation_prob
                j = randi(n_qubits);
                theta(i, j) = theta(i, j) + pi;
            end
        end
        
        % Wrap theta to [0, 2*pi)
        theta = mod(theta, 2 * pi);
        
        % Compute new positions
        for i = 1:N
            psi = compute_amplitudes(theta(i, :));
            psi = psi(1:Dim);
            X(i, :) = LB + range .* (psi' + 1) / 2;
            Fitness(i) = F_obj(X(i, :));
        end
        
        % Update personal and global bests
        for i = 1:N
            if Fitness(i) < pbest_fitness(i)
                pbest_fitness(i) = Fitness(i);
                pbest_theta(i, :) = theta(i, :);
            end
            if Fitness(i) < gbest_fitness
                gbest_fitness = Fitness(i);
                gbest_theta = theta(i, :);
                gbest = X(i, :);
            end
        end
        
        % Stagnation handling
        if gbest_fitness >= old_gbest_fitness
            stagnation_count = stagnation_count + 1;
        else
            stagnation_count = 0;
        end
        if stagnation_count >= stagnation_limit
            reset_count = round(0.25 * N);  % Adjusted reset fraction for more refresh
            [~, worst_idx] = sort(Fitness, 'descend');
            for k = 1:reset_count
                theta(worst_idx(k), :) = 2 * pi * rand(1, n_qubits);
                psi = compute_amplitudes(theta(worst_idx(k), :));
                psi = psi(1:Dim);
                X(worst_idx(k), :) = LB + range .* (psi' + 1) / 2;
                Fitness(worst_idx(k)) = F_obj(X(worst_idx(k), :));
            end
            stagnation_count = 0;
        end
        
        % Store
        Curve(it) = gbest_fitness;
        PopHistRun{it} = X;
    end
    
    % Final local search refinement on best solution (requires Optimization Toolbox)
    options = optimset('Display', 'off', 'MaxIter', 50, 'TolX', 1e-8, 'TolFun', 1e-8);
    [BestSol, BestFitness] = fminsearch(F_obj, gbest, options);
    
end
function psi = compute_amplitudes(theta)
    n = length(theta);
    psi = 1;
    for j = 1:n
        q = [cos(theta(j)); sin(theta(j))];
        psi = kron(psi, q);
    end
end
