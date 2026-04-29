# Optimization Techniques 

## Overview

This document contains MATLAB implementations of four fundamental optimization methods commonly used in operations research and linear programming:

- **Simplex Method**: Standard algorithm for solving linear programming problems
- **Big M Method**: Extension of the simplex method for problems with equality and greater-than constraints
- **Least Cost Method**: Heuristic for solving transportation problems
- **Steepest Descent Method**: Gradient-based optimization for unconstrained minimization

---

## 1. Simplex Method

**Objective**: Minimize z = x₁ - 3x₂ + 2x₃

**Subject to**:

- 3x₁ - x₂ + 2x₃ ≤ 7
- -2x₁ + 4x₂ ≤ 12
- -4x₁ + 3x₂ + 8x₃ ≤ 10
- x₁, x₂, x₃ ≥ 0

```matlab
clc
clear all
format short

C=[-1 3 -2];
info=[3 -1 2; -2 4 0; -4 3 8];
b=[7; 12; 10];
NOVariables=size(info,2);
s=eye(size(info,1));
A=[info s b];
Cost=zeros(1,size(A,2));
Cost(1:NOVariables)=C;

BV=NOVariables+1:size(A,2)-1;

ZRow=Cost(BV)*A-Cost;

ZjCj=[ZRow;A];
SimpTable=array2table(ZjCj);
SimpTable.Properties.VariableNames(1:size(ZjCj,2))={'x_1','x_2','x_3','s_1','s_2','s_3','Sol'};

Run=true;
while Run
    if any(ZRow<0)
        fprintf('The current BFS is not Optimal \n')
        fprintf('\n ============The Next Iteration Results========\n')
        disp('Old Basic Variable (BV)=')
        disp(BV)

        ZC=ZRow(1:end-1);
        [EnterCol,Pvt_Col]=min(ZC);
        fprintf('The most negative element in ZRow is %d Corresponding to Column %d \n', EnterCol, Pvt_Col)
        fprintf('Entering Variable is %d \n', Pvt_Col)

        sol=A(:,end);
        Column=A(:,Pvt_Col);
        if all(Column<=0)
            error('LPP has unbounded solution. All entries <=0 in Column %d', Pvt_Col)
        else
            for i=1:size(Column,1)
                if Column(i)>0
                    ratio(i)=sol(i)./Column(i);
                else
                    ratio(i)=inf;
                end
            end
            [MinRatio, Pvt_Row]=min(ratio);
            fprintf('Minimum ratio corresponding to pivot row is %d \n',Pvt_Row)
            fprintf('Leaving Variable is %d \n', BV(Pvt_Row))
        end

        BV(Pvt_Row)=Pvt_Col;
        disp('New Basic Variables (BV) =')
        disp(BV)

        Pvt_Key=A(Pvt_Row,Pvt_Col);
        A(Pvt_Row,:)=A(Pvt_Row,:)./Pvt_Key;
        for i=1:size(A,1)
            if i~=Pvt_Row
                A(i,:)=A(i,:)-A(i,Pvt_Col).*A(Pvt_Row,:);
            end
            ZRow=ZRow-ZRow(Pvt_Col).*A(Pvt_Row,:);
            ZjCj=[ZRow;A];
            SimpTable=array2table(ZjCj);
            SimpTable.Properties.VariableNames(1:size(ZjCj,2))={'x_1','x_2','x_3','s_1','s_2','s_3','Sol'};
        end

        BFS=zeros(1,size(A,2));
        BFS(BV)=A(:,end);
        BFS(end)=sum(BFS.*Cost);
        CurrentBFS=array2table(BFS);
        CurrentBFS.Properties.VariableNames(1:size(CurrentBFS,2))={'x_1','x_2','x_3','s_1','s_2','s_3','Sol'};
    else
        Run=false;
        fprintf('======********************============\n')
        fprintf('The current BFS is optimal and Optimality is reached \n')
        fprintf('======********************============\n')
    end
end
```

---

## 2. Big M Method

**Objective**: Minimize z = 2x₁ + x₂

**Subject to**:

- 3x₁ + x₂ = 3
- 4x₁ + 3x₂ ≥ 6
- x₁ + 2x₂ ≤ 3
- x₁, x₂ ≥ 0

```matlab
clc
clear all
format short

Variables={'x_1','x_2','s_2','s_3','A_1','A_2','Sol'};
M=1000;
Cost=[-2,-1,0,0,-M,-M,0];
A=[3, 1, 0, 0, 1, 0, 3;4, 3, -1, 0, 0, 1, 6;1, 2, 0, 1, 0, 0, 3];
s=eye(size(A,1));

BV=[];
for j=1:size(s,2)
    for i=1:size(A,2)
        if A(:,i)==s(:,j)
            BV=[BV i];
        end
    end
end

ZjCj=Cost(BV)*A-Cost;

ZCj=[ZjCj;A];
SimpTable=array2table(ZCj);
SimpTable.Properties.VariableNames(1:size(ZCj,2))=Variables;

Run=true;
while Run
    ZC=ZjCj(:,1:end-1);
    if any(ZC<0)
        fprintf('The current BFS is not Optimal \n')
        fprintf('\n ============The Next Iteration Results========\n')

        [Entval,Pvt_Col]=min(ZC);
        fprintf('Entering Column is %d \n', Pvt_Col)

        sol=A(:,end);
        Column=A(:,Pvt_Col);
        if all(Column<=0)
            fprintf('LPP has unbounded solution')
        else
            for i=1:size(Column,1)
                if Column(i)>0
                    ratio(i)=sol(i)./Column(i);
                else
                    ratio(i)=inf;
                end
            end
            [MinRatio, Pvt_Row]=min(ratio);
            fprintf('Leaving Row is %d \n', Pvt_Row)
        end

        BV(Pvt_Row)=Pvt_Col;
        Pvt_Key=A(Pvt_Row,Pvt_Col);
        A(Pvt_Row,:)=A(Pvt_Row,:)./Pvt_Key;
        for i=1:size(A,1)
            if i~=Pvt_Row
                A(i,:)=A(i,:)-A(i,Pvt_Col).*A(Pvt_Row,:);
            end
            ZjCj=ZjCj-ZjCj(Pvt_Col).*A(Pvt_Row,:);
            ZCj=[ZjCj;A];
            SimpTable=array2table(ZCj);
            SimpTable.Properties.VariableNames(1:size(ZCj,2))=Variables;
        end
    else
        Run=false;
        fprintf('======********************============\n')
        fprintf('The current BFS is optimal and Optimality is reached \n')
        fprintf('======********************============\n')
    end
end

BFS=zeros(1,size(A,2));
BFS(BV)=A(:,end);
BFS(end)=sum(BFS.*Cost);
CurrentBFS=array2table(BFS);
CurrentBFS.Properties.VariableNames(1:size(CurrentBFS,2))=Variables;
```

---

## 3. Least Cost Method

**Transportation Problem**: Allocate supplies to demands with minimum cost

```matlab
clc
clear all
format short

Cost=[11 20 7 8; 21 16 10 12; 8 12 18 9];
A=[50 40 70];
B=[30 25 35 40];

if sum(A)==sum(B)
    fprintf('Given Transportation Problem is Balanced \n')
else
    fprintf('Given Transportation Problem is Unbalanced \n')
    if sum(A)<sum(B)
        Cost(end+1,:)=zeros(1,size(B,2));
        A(end+1)=sum(B)-sum(A);
    elseif sum(B)<sum(A)
        Cost(:,end+1)=zeros(1,size(A,2));
        B(end+1)=sum(A)-sum(B);
    end
end

ICost=Cost;
X=zeros(size(Cost));
[m,n]=size(Cost);
BFS=m+n-1;

for i=1:size(Cost,1)
    for j=1:size(Cost,2)
        hh=min(Cost(:));
        [Row_index, Col_index]=find(hh==Cost);
        x11=min(A(Row_index),B(Col_index));
        [Value,index]=max(x11);
        ii=Row_index(index);
        jj=Col_index(index);
        y11=min(A(ii),B(jj));
        X(ii,jj)=y11;
        A(ii)=A(ii)-y11;
        B(jj)=B(jj)-y11;
        Cost(ii,jj)=Inf;
    end
end

fprintf('Initial BFS =\n')
IBFS=array2table(X);
disp(IBFS)

TotalBFS=length(nonzeros(X));
if TotalBFS==BFS
    fprintf('Initial BFS is Non-Degenerate \n')
else
    fprintf('Initial BFS is Degenerate \n')
end

InitialCost=sum(sum(ICost.*X));
fprintf('Initial BFS Cost is = %d \n',InitialCost)
```

---

## 4. Steepest Descent Method

**Objective**: Minimize f(x₁, x₂) = x₁² + 2x₂²

```matlab
clear all;
clc;

f = @(x1, x2) x1.^2 + 2*x2.^2;
grad_f = @(x1, x2) [2*x1; 4*x2];

x = [3; 3];

max_iter = 100;
tol = 1e-6;
alpha = 0.1;

fprintf('Initial point: (%f, %f)\n', x(1), x(2));

for iter = 1:max_iter
    gradient = grad_f(x(1), x(2));

    if norm(gradient) < tol
        fprintf('Converged after %d iterations\n', iter-1);
        break;
    end

    x = x - alpha * gradient;

    fprintf('Iteration %d: x = (%f, %f), f(x) = %f\n', iter, x(1), x(2), f(x(1), x(2)));
end

fprintf('Final solution: (%f, %f)\n', x(1), x(2));
fprintf('Minimum function value: %f\n', f(x(1), x(2)));
```

---

## Summary

This document provides clean, organized MATLAB implementations of four key optimization techniques used in operations research and numerical optimization. Each method is presented with its corresponding problem formulation and code implementation.
