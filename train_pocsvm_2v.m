function model = train_pocsvm_2v(x, x2, y, kerType, nv_a, nv_b, C, g4kerA, g4kerB, gamma4psvm, epsilon)
    if (nargin < 11)
        model.epsilon=0.01;
    else
        model.epsilon = epsilon;
    end

    n = length(y);
    model.kerType=kerType;
    model.g4kerA = g4kerA;
    model.g4kerB = g4kerB;
    model.gamma4psvm=gamma4psvm;
    model.nv_a=nv_a;
    model.nv_b=nv_b;
    model.C=C;
    Ca = 1/(nv_a*n);
    Cb = 1/(nv_b*n);

    options = optimset; 
    options.Display = 'off'; 
    
    temp1=kernel(x,x,kerType,g4kerA);
    temp2=(1/gamma4psvm)*kernel(x2,x2,kerType,g4kerB); 
    
    H1=temp1; % H1
    H2=temp2; % H2
    H3=temp1+temp2; % H3
    
    H=[H1,zeros(n,n),-H1,H1,zeros(n,n),-H1;
        zeros(n,n),H2,H2,-H2,-H2,zeros(n,n);
        -H1,H2,H3,-H3',-H2,H1; 
        H1,-H2,-H3,H3,H2,-H1;
        zeros(n,n),-H2,-H2,H2,H2,zeros(n,n);
        -H1,zeros(n,n),H1,-H1,zeros(n,n),H1];
    
    f=[zeros(2*n,1);epsilon*ones(2*n,1);zeros(2*n,1)]; 

    A = [eye(n),zeros(n,3*n),eye(n),zeros(n,n); 
         zeros(n,n),eye(n),zeros(n,3*n),eye(n);
         zeros(n,2*n),eye(n),eye(n),zeros(n,2*n)];
    b = [Ca*ones(n,1);Cb*ones(n,1);C*ones(n,1)]; 
    Aeq = [ones(1,n), zeros(1,n), -ones(1,n), ones(1,n), zeros(1,n), -ones(1,n);
           zeros(1,n), ones(1,n), ones(1,n), -ones(1,n), -ones(1,n), zeros(1,n)];
    beq = [1;1];
    lb = zeros(6*n,1);
    ub = [Ca*ones(n,1);Cb*ones(n,1);C*ones(2*n,1);Ca*ones(n,1);Cb*ones(n,1)];
    a0 = zeros(6*n,1);

    [a]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options); % deal weith qp problem
    alpha_a=a(1:n);
    alpha_b=a(n+1:2*n);
    beta_a=a(2*n+1:3*n); % beta_a  beat^+
    beta_b=a(3*n+1:4*n); % beta_a  beat^-
    lamda_a=a(4*n+1:5*n);
    lamda_b=a(5*n+1:6*n);
    model.Wa = alpha_a - beta_a + beta_b - lamda_b;
    model.Wb = (1/gamma4psvm)*(alpha_b + beta_a - beta_b - lamda_a);
    model.x=x;
    model.x2=x2;
    model.y=y;

%% linprog to calculate rho 
    kA = kernel(x, x, model.kerType, model.g4kerA) * model.Wa; 
    kB = kernel(x2, x2, model.kerType, model.g4kerB) * model.Wb;
    cA = 1/(model.nv_a*n);
    cB = 1/(model.nv_b*n);
    f = [-1; -1; cA.*ones(n, 1); cB.*ones(n, 1); model.C.*ones(n, 1)];
    
    A = [ones(n,1), zeros(n,1), -eye(n), zeros(n,n), zeros(n,n);
       zeros(n,1), ones(n,1), zeros(n,n), -eye(n), zeros(n,n);
       -ones(n,1), ones(n,1), zeros(n,n), zeros(n,n), -eye(n);
       ones(n,1), -ones(n,1), zeros(n,n), zeros(n,n), -eye(n);
       zeros(n,1), -ones(n,1), -eye(n), zeros(n,n), zeros(n,n);
       -ones(n,1), zeros(n,1), zeros(n,n), -eye(n), zeros(n,n);
       zeros(n,1), zeros(n,1), -eye(n), zeros(n,n), zeros(n,n);
       zeros(n,1), zeros(n,1), zeros(n,n), -eye(n), zeros(n,n);
       zeros(n,1), zeros(n,1), zeros(n,n), zeros(n,n), -eye(n)];
    b = [kA; kB; (-kA + kB + model.epsilon); (kA - kB + model.epsilon); -kB; -kA; zeros(3*n,1)];

    myresult = linprog(f, A, b, [], [], [], [], options);
    model.rA = myresult(1);
    model.rB = myresult(2);
end
