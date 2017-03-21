import clusters
wants, people, data = clusters.readfile('zebo.txt')
clust = clusters.hcluster(data, distance=clusters.taniomoto)
clusters.drawdendrogram(clust, wants)