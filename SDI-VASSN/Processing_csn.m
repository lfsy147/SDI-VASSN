clc;
clear;
a=importdata('./stMVC_test_data/DLPFC_151673/stMVC/gene_output.csv');
new_gene_expression_data=a.data;
Final_genes=a.textdata(2:end,:);
new_gene_expression_data(isnan(new_gene_expression_data))=0;
for i=1:size(new_gene_expression_data,2)
    
    
    i;
    net=csnet(new_gene_expression_data,i,0.01,0.1,1);
    A_net=net{1,i};
    k_degree=sum(A_net,2);
    [aa,bb]=sort(k_degree,'descend');
    Sample_selected_genes{i,1}=Final_genes(bb(1:50));%result
    
end
xlswrite('./stMVC_test_data/DLPFC_151673/stMVC/output_gene_name.csv', Sample_selected_genes);

clc;clear;
table1 = readtable('./stMVC_test_data/DLPFC_151673/stMVC/gene_output.csv');
table2 = readtable('./stMVC_test_data/DLPFC_151673/stMVC/output_gene_name.csv');

data1=table2cell(table1);
data2 = table2{:,:};


for i = 1:size(data2, 1)
    indices = data2(i, :);
    [dif,indices_diff] = setdiff(data1(:, 1), indices);
    data1(indices_diff(:), i+1) = num2cell(0);
end

dataTable = cell2table(data1);
writetable(dataTable, './stMVC_test_data/DLPFC_151673/stMVC/csn_output.csv');





