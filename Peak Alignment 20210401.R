#' @name peakAlignment
#' @author Fujian Zheng <zhengfj@dicp.ac.cn>
#' @description peak alignment
#' @param file.path
#' @param ppm
#' @param absMz
#' @param minFraction
#' @param minSamples
#' @return peakAlignmentRes
#' @example peakAlignmentRes <- peakAlignment(file.path = "E:/FDS/test",
#'                                            ppm = 5,
#'                                            absMz = 0.001,
#'                                            minFraction = 0,
#'                                            minSamples = 1)

peakAlignment <- function(file.path,ppm,absMz,minFraction,minSamples){
  library(xcms)
  library(openxlsx)
  library(stringr)
  
  setwd(file.path)
  files <- dir()
  
  peak.data <- na.omit(read.xlsx(xlsxFile = files[1]))
  print(files[1])
  if (peak.data[1,1] == "m/z"){
    colnames(peak.data) <- peak.data[1,]
    peak.data <- peak.data[-1,]
  }
  mz.data <- as.numeric(peak.data$`m/z`)
  rt.data <- rep(1,length(mz.data))
  int.data <- as.numeric(peak.data$Intensity)
  
  # peak.detection.df <- as.data.frame(do_findPeaks_MSW(mz.data, int.data, snthresh = snthresh, amp.Th = amp.Th))
  peak.detection.df <- as.data.frame(cbind(mz.data,rt.data,int.data))
  colnames(peak.detection.df) <- c("mz", "rt", "Intensity")
  peak.detection.df$sample <- 1
  sample.groups.name <- strsplit(files[1],split = ".xlsx")[[1]]
  
  for (i in c(2:length(files))){
    peak.data <- na.omit(read.xlsx(xlsxFile = files[i]))
    print(files[i])
    if (peak.data[1,1] == "m/z"){
      colnames(peak.data) <- peak.data[1,]
      peak.data <- peak.data[-1,]
    }
    mz.data <- as.numeric(peak.data$`m/z`)
    rt.data <- rep(1,length(mz.data))
    int.data <- as.numeric(peak.data$Intensity)
    
    # peak.detection.df.i <- as.data.frame(do_findPeaks_MSW(mz.data, int.data, snthresh = snthresh, amp.Th = amp.Th))
    peak.detection.df.i <- as.data.frame(cbind(mz.data,rt.data,int.data))
    colnames(peak.detection.df.i) <- c("mz", "rt", "Intensity")
    peak.detection.df.i$sample <- i
    sample.groups.name.i <- strsplit(files[i],split = ".xlsx")[[1]]
    sample.groups.name <- c(sample.groups.name,sample.groups.name.i)
    peak.detection.df <- rbind(peak.detection.df, peak.detection.df.i)
  }
  
  ans <- do_groupPeaks_mzClust(peaks = peak.detection.df, sampleGroups = c(1:length(files)), 
                               ppm = ppm, absMz = absMz,
                               minFraction = minFraction, minSamples = minSamples)
  
  # ans2 <- do_groupChromPeaks_density(peaks = peak.detection.df, sampleGroups = c(1:length(files)), bw = 30,
  #                            minFraction = 0.5, minSamples = 1, binSize = 0.25,
  #                            maxFeatures = 50, sleep = 0)
  
  peak.table <- as.data.frame(ans$featureDefinitions)
  peak.table[,(8+length(files)):(7+2*length(files))] <- 0
  colnames(peak.table)[(8+length(files)):(7+2*length(files))] <- sample.groups.name
  for (i in c(1:nrow(peak.table))){
    index.i <- ans$peakIndex[[i]]
    index.i.peak.table <- peak.detection.df[index.i,]
    for (j in c(1:length(files))){
      if (length(which(index.i.peak.table$sample == j)) == 1){
        peak.table[i,((7+length(files))+j)] <- index.i.peak.table$Intensity[which(index.i.peak.table$sample == j)]
      }
      else if (length(which(index.i.peak.table$sample == j)) > 1){
        peak.table[i,((7+length(files))+j)] <- sum(index.i.peak.table$Intensity[which(index.i.peak.table$sample == j)])
      }
    }
  }
  peakAlignmentRes <- list(nonalign = peak.detection.df, align = peak.table)
  return(peakAlignmentRes)
}

peakAlignmentRes <- peakAlignment(file.path = "D:/1808/Data/fds-20230228/Data/PC-3 all/Data",
                                  ppm = 8,
                                  absMz = 0,
                                  minFraction = 0.5,
                                  minSamples = 564)
write.csv(peakAlignmentRes$align,file = "D:/1808/Data/fds-20230228/Data/PC-3 all/PC3-all-20230414.csv")

