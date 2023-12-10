# monocle3 -----------------------------------------------------------------
library(monocle3)
library(dplyr)
library(ggplot2)
packageVersion('monocle3')
library(reticulate)

#set wd
rm(list = ls())
setwd("D:/PyProject/monocle")

#read data, read.delim
GM_expr_matrix <- read.table("HSMM CP2M 20230918/GM_expr_matrix.csv",row.names = 1,header = T, sep = ",", check.names=FALSE)
GM_sample_sheet <- read.delim("HSMM CP2M 20230918/GM_sample_sheet.csv",row.names = 1,header = T,sep = ",")
GM_gene_annotation <- read.delim("HSMM CP2M 20230918/GM_gene_annotation.csv",row.names = 1,header = T,sep = ",")

#look data；
dim(GM_expr_matrix)
GM_expr_matrix[1:6,1:7]
head(GM_gene_annotation)
head(GM_sample_sheet)

# Create monocle3 CDS(CellDataSet)object
GM <- new_cell_data_set(as.matrix(GM_expr_matrix),
                     cell_metadata = GM_sample_sheet,
                     gene_metadata = GM_gene_annotation)
GM
GM <- preprocess_cds(GM, num_dim = 100)
p1 = plot_pc_variance_explained(GM)
show(p1)
ggsave("HSMM CP2M 20230918/pc_variance_explained_GM.tiff", plot = p1,width = 12,height = 8,units = c("cm"),dpi = 300)

GM <- reduce_dimension(GM, preprocess_method = "PCA",reduction_method = c("UMAP"))
table(colData(GM)$"cell_type")
# color based on cell type
p2 = plot_cells(GM, color_cells_by="cell_type")
show(p2)
ggsave("HSMM CP2M 20230918/UMAP_GM.tiff", plot = p2,width = 8,height = 8,units = c("cm"),dpi = 300)
# (except cao_cell_type, can also use below)
colnames(colData(GM))
p3 = plot_cells(GM, genes=c("m/z 132.0769", "m/z 184.0736", "m/z 203.2233",
                            "m/z 181.0836", "m/z 163.0757", "m/z 198.0974"))
show(p3)
ggsave("HSMM CP2M 20230918/UMAP_GlucoseGMraw.tiff", plot = p3,width = 12,height = 8,units = c("cm"),dpi = 300)

GM <- reduce_dimension(GM, reduction_method="tSNE")
# t-SNE 
p4 = plot_cells(GM, reduction_method="tSNE", color_cells_by="cell_type")
show(p4)
ggsave("HSMM CP2M 20230918/t-SNEGM.tiff", plot = p4,width = 8,height = 8,units = c("cm"),dpi = 300)

#leiden clustering
#如果resolution不为空且cluster_method = "louvain"，则会发出一条警告消息并将cluster_method="leiden"。
GM <- cluster_cells(GM, reduction_method = "UMAP", cluster_method="leiden", resolution=c(10^seq(-6,-1)))
p5 = plot_cells(GM)
show(p5)
ggsave("HSMM CP2M 20230918/UMAPclusterGM.tiff", plot = p5,width = 8,height = 8,units = c("cm"),dpi = 300)
# save cluster label to csv
write.csv(p5[["data"]], "HSMM CP2M 20230918/UMAPleidenclusterGM.csv", row.names = TRUE)


#Identify genes that are differentially expressed between clusters by Jensen-Shannon distance
#http://cole-trapnell-lab.github.io/monocle-release/monocle3/
#################################################################

#GM <- learn_graph(GM)
#start <- Sys.time()
#spatial_res <- graph_test(GM, neighbor_graph="principal_graph", reduction_method = "UMAP",k = 25, 
#                          method = c('Moran_I'), cores = 8, verbose = FALSE)
#end <- Sys.time()
#end - start
#cluster_marker_res <- find_cluster_markers(GM, spatial_res, group_by = 'Cluster', morans_I_threshold = 0.25)
#genes <- (cluster_marker_res %>%
#            dplyr::filter(mean > 0.5, percentage > 0.1) %>%
#            dplyr::group_by(Group) %>% dplyr::slice(which.max(specificity)))
#options(repr.plot.width=22, repr.plot.height=12)
#p17 = plot_markers_by_group(GM, genes$gene_short_name, group_by = 'Cluster', ordering_type = 'maximal_on_diag')
#show(p17)
#ggsave("HSMM CP2M 20230918/Jensen-Shannon-distanceclustermarker.tiff", plot = p17, width = 16, height = 16, units = c("cm"),dpi = 300)


#Visualizing marker expression across cell clusters
#genes <-
#  (cluster_marker_res %>%
#     dplyr::filter(mean > 0.5, percentage > 0.1) %>%
#     dplyr::group_by(Group) %>%
#     dplyr::top_n(5, wt = specificity))
#p18 = plot_markers_cluster(GM, as.character(genes$gene_short_name),
#                     minimal_cluster_fraction = 0.05)
#show(p18)
#ggsave("HSMM CP2M 20230918/heatmap-markers-cellclusters.tiff", plot = p18, width = 16, height = 16, units = c("cm"),dpi = 300)

#Identify and plot marker genes for each cluster
#################################################
marker_test_res = top_markers(GM, group_cells_by="cluster", marker_sig_test = TRUE, 
                              reference_cells=500, cores=8)
# 设置reference_cells是随机挑出来这些数量的细胞作为参照，然后让top_markers和参照集中的基因进行显著性检验；
#另外reference_cells还可以是来自colnames(cds)的细胞名
marker_test_res[1:4,1:4]

top_specific_markers = marker_test_res %>%
  filter(fraction_expressing >= 0.10) %>%
  group_by(cell_group) %>%
  top_n(1, pseudo_R2)
# 基因id去重
top_specific_marker_ids = unique(top_specific_markers %>% pull(gene_id))
# ordering_type = c("cluster_row_col", "maximal_on_diag", "none")
# 'cluster_row_col' (use biclustering to cluster the rows and columns)
# 'maximal_on_diag' (position each column so that the maximal color shown on each column on the diagonal, 
# if the current maximal is used in earlier columns, the next largest one is position)
#  'none' (preserve the ordering from the input gene or alphabetical ordering of groups)
p15 = plot_genes_by_group(GM,
                         top_specific_marker_ids,
                         group_cells_by="cluster",
                         ordering_type="maximal_on_diag",
                         max.size=3)
show(p15)
ggsave("HSMM CP2M 20230918/UMAPclustermarkerGM.tiff", plot = p15, width = 16, height = 16, units = c("cm"),dpi = 300)

#Plot expression for one or more genes as a violin plot
###########################################################
chosed_metabolites <- c("M413", "M795", "M650", "M160", "M235", "M415")

GM_subset <- GM[rowData(GM)$gene_short_name %in% chosed_metabolites,]
# 提取p5[["data"]]中每行样品对应的cell_group
cell_group_values <- p5[["data"]]$cell_group
# 将cell_group_values加入到GM_subset中
GM_subset$cell_group <- cell_group_values

p19 = plot_genes_violin(GM_subset, group_cells_by="cell_group", ncol=2, normalize = TRUE,log_scale = TRUE) +
  theme(axis.text.x=element_text(angle=45, hjust=1))
show(p19)
ggsave("HSMM CP2M 20230918/UMAPclusterviolin.tiff", plot = p19, width = 16, height = 16, units = c("cm"),dpi = 300)

#inspect more markers, change top_n
top_specific_markers = marker_test_res %>%
  filter(fraction_expressing >= 0.10) %>%
  group_by(cell_group) %>%
  top_n(3, pseudo_R2)

top_specific_marker_ids = unique(top_specific_markers %>% pull(gene_id))

p16 = plot_genes_by_group(GM,
                          top_specific_marker_ids,
                          group_cells_by="cluster",
                          ordering_type="cluster_row_col",
                          max.size=3)
show(p16)
ggsave("HSMM CP2M 20230918/UMAPcluster3markerGM.tiff", plot = p16,width = 18, height = 18,units = c("cm"),dpi = 300)

###################################################################################



# extract clusters from CDS object
clusters(GM) 
#number of cluster

p6 = plot_cells(GM, color_cells_by="partition", group_cells_by="partition")
show(p6)
ggsave("HSMM CP2M 20230918/UMAPpartitionGM.tiff", plot = p6,width = 8,height = 8,units = c("cm"),dpi = 300)
# save partition label to csv
write.csv(p6[["data"]], "HSMM CP2M 20230918/UMAPpartitionGM.csv", row.names = TRUE)
#################################################################


# number of partitions
GM@clusters$UMAP$partitions
p7 = plot_cells(GM, color_cells_by="cell_type")
show(p7)
ggsave("HSMM CP2M 20230918/UMAPpartitionTRuelableGM.tiff", plot = p7,width = 8,height = 8,units = c("cm"),dpi = 300)

p8 = plot_cells(GM, color_cells_by="cell_type", label_groups_by_cluster=FALSE)
show(p8)
ggsave("HSMM CP2M 20230918/UMAPpartitionTruegroupGM.tiff", plot = p8,width = 8,height = 8,units = c("cm"),dpi = 300)

marker_test_res = top_markers(GM, group_cells_by="partition", reference_cells=500, cores=8)
# 设置reference_cells是随机挑出来这些数量的细胞作为参照，然后让top_markers和参照集中的基因进行显著性检验；
#另外reference_cells还可以是来自colnames(cds)的细胞名
marker_test_res[1:4,1:4]

top_specific_markers = marker_test_res %>%
  filter(fraction_expressing >= 0.10) %>%
  group_by(cell_group) %>%
  top_n(1, pseudo_R2)
# 基因id去重
top_specific_marker_ids = unique(top_specific_markers %>% pull(gene_id))
p9 = plot_genes_by_group(GM,
                    top_specific_marker_ids,
                    group_cells_by="partition",
                    ordering_type="maximal_on_diag",
                    max.size=3)
show(p9)
ggsave("HSMM CP2M 20230918/UMAPpartitionmarkerGM.tiff", plot = p9,width = 12,height = 12,units = c("cm"),dpi = 300)

#Plot expression for one or more genes as a violin plot
###########################################################
chosed_metabolites <- c("M413", "M795", "M650", "M160", "M235", "M415")
GM_subset <- GM[rowData(GM)$gene_short_name %in% chosed_metabolites,]
# 提取p6[["data"]]中每行样品对应的cell_group即partition
cell_group_values <- p6[["data"]]$cell_group
# 将cell_group_values加入到GM_subset中
GM_subset$cell_group <- cell_group_values

p20 = plot_genes_violin(GM_subset, group_cells_by="cell_group", ncol=2) +
  theme(axis.text.x=element_text(angle=45, hjust=1))
show(p20)
ggsave("HSMM CP2M 20230918/UMAPpartitionviolin.tiff", plot = p20, width = 16, height = 16, units = c("cm"),dpi = 300)

#inspect more markers, change top_n
top_specific_markers = marker_test_res %>%
  filter(fraction_expressing >= 0.10) %>%
  group_by(cell_group) %>%
  top_n(3, pseudo_R2)

top_specific_marker_ids = unique(top_specific_markers %>% pull(gene_id))

p10 = plot_genes_by_group(GM,
                    top_specific_marker_ids,
                    group_cells_by="partition",
                    ordering_type="cluster_row_col",
                    max.size=3)
show(p10)
ggsave("HSMM CP2M 20230918/UMAPpartition3markerGM.tiff", plot = p10,width = 12, height = 12,units = c("cm"),dpi = 300)

GM <- learn_graph(GM)
p11 = plot_cells(GM,
           color_cells_by = "cell_type",
           label_groups_by_cluster=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE)
show(p11)
ggsave("HSMM CP2M 20230918/learntypeGM.tiff", plot = p11,width = 8,height = 8,units = c("cm"),dpi = 300)

p12 = plot_cells(GM,
           color_cells_by = "Hours",
           label_cell_groups=FALSE,
           label_leaves=TRUE,
           label_branch_points=TRUE,
           graph_label_size=1.5)
show(p12)
ggsave("HSMM CP2M 20230918/HoursGM.tiff", plot = p12,width = 8,height = 8,units = c("cm"),dpi = 300)

# 官方给出了一个函数，这里定义了一个time_bin，选择了最早的时间点区间。
get_earliest_principal_node <- function(GM, time_bin="14"){
  # 首先找到出现在最早时间区间的细胞ID
  cell_ids <- which(colData(GM)[, "Hours"] == time_bin)
  
  closest_vertex <-
    GM@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex
  closest_vertex <- as.matrix(closest_vertex[colnames(GM), ])
  root_pr_nodes <-
    igraph::V(principal_graph(GM)[["UMAP"]])$name[as.numeric(names
                                                              (which.max(table(closest_vertex[cell_ids,]))))]
  
  root_pr_nodes
}
GM = order_cells(GM, root_pr_nodes=get_earliest_principal_node(GM))

p13 = plot_cells(GM,
           color_cells_by = "pseudotime",
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           graph_label_size=1.5)
show(p13)
ggsave("HSMM CP2M 20230918/pseudotimeGM.tiff", plot = p13,width = 8,height = 8,units = c("cm"),dpi = 300)

# 3D trajectories
GM_3d = reduce_dimension(GM, max_components = 3)
GM_3d = cluster_cells(GM_3d)
GM_3d = learn_graph(GM_3d)
GM_3d = order_cells(GM_3d, root_pr_nodes=get_earliest_principal_node(GM))

p14 = GM_3d_plot_obj = plot_cells_3d(GM_3d, color_cells_by="cell_type")
GM_3d_plot_obj
ggsave("HSMM CP2M 20230918/pseudotime3dGM.tiff", plot = GM_3d_plot_obj,width = 12,height = 8,units = c("cm"),dpi = 300)



