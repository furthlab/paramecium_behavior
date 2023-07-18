#' Plot Shock Data
#'
#' This function plots shock data along with average swimming velocity and
#' generates a bar graph showing the grand mean with confidence intervals.
#'
#' @param vec Numeric array of swimming velocity data rows are cells and colummns are video frames.
#' @param fps Numeric value indicating frames per second.
#' @param shock_on_times Numeric vector of shock on times in seconds.
#' @param shock_off_times Numeric vector of shock off times in seconds.
#'
#' @importFrom graphics layout plot lines box polygon par
#' @importFrom stats apply mean
#' @import sciplot
#'
#' @examples
#' dat<-read.table('~/Downloads/track_data.csv', sep=',', header=TRUE)
#' vec<-lapply(unique(dat$ID), function(x){sqrt(diff(dat$X[dat$ID==x])^2+diff(dat$Y[dat$ID==x])^2)  })
#' vec<-do.call("rbind", vec)
#'
#' fps<-23.977351668359663  # Replace with actual frames per second
#'
#' shock_on_times <- c(2.8, 16, 28.2, 41.75)  # Replace with actual shock on times in seconds
#' shock_off_times <- c(6.7, 19, 33, 44.5)  # Replace with actual shock off times in seconds
#'
#' out<-plot_shock_data(vec, fps, shock_on_times, shock_off_times)
#'
#' @export
plot_shock_data <- function(vec, fps, shock_on_times, shock_off_times, col='pink', event='Shock', remove=2, ymax=NULL) {
  layout(matrix(c(1,1,1,2,1,1,1,2), 2, 4, byrow = TRUE))
  speed<-(apply(vec, 2, mean, na.rm=TRUE))[-c(1:remove)]
  if(is.null(ymax)){
    ymax<-max((apply(vec, 2, mean, na.rm=TRUE))[-c(1:remove)])*1.2
  }

  plot((1:ncol(vec))[-c(1:remove)]/fps, (apply(vec, 2, mean, na.rm=TRUE))[-c(1:remove)],
       type = 'l', lwd = 1, ylim = c(0, ymax), ylab = "Mean swimming velocity", las = 1, xlab = "Time (seconds)")

  for (i in 1:length(shock_on_times)) {
    polygon(c(c(shock_on_times[i], shock_off_times[i]), rev(c(shock_on_times[i], shock_off_times[i]))), c(-10, -10, 20, 20), col = col)
  }

  box()
  lines((1:ncol(vec))[-c(1:remove)]/fps, (apply(vec, 2, mean, na.rm=TRUE))[-c(1:remove)])
  lines( smooth.spline((1:ncol(vec))[-c(1:remove)]/fps, (apply(vec, 2, mean, na.rm=TRUE))[-c(1:remove)]), col='red')

  tim<-(1:ncol(vec))[-c(1:remove)]/fps
  shockon <- c()
  for (i in 1:length(shock_on_times)) {
    shockon[i] <- mean(speed[which(tim > shock_on_times[i] & tim < shock_off_times[i])], na.rm=TRUE)
    lines(c(shock_on_times[i], shock_off_times[i]), c(shockon[i], shockon[i]), lwd = 4, col = 'blue')
  }

  if(any(is.na(shockon))){
    shockon[is.na(shockon)]<-mean(shockon, na.rm = TRUE)
  }

  shockoff <- mean(speed[which(tim < shock_on_times[1])], na.rm=TRUE)
  lines(c(0, shock_on_times[1]), c(shockoff, shockoff), lwd = 4, col = 'blue')
  for (i in 2:length(shock_on_times)) {
    shockoff[i] <- mean(speed[which(tim < shock_on_times[i] & tim > shock_off_times[i-1])], na.rm=TRUE)
    lines(c(shock_off_times[i-1], shock_on_times[i]), c(shockoff[i], shockoff[i]), lwd = 4, col = 'blue')
  }
  shockoff[i+1] <- mean(speed[which(tim > shock_off_times[length(shock_off_times)])], na.rm=TRUE)
  lines(c(shock_off_times[i], max(tim)), c(shockoff[i+1], shockoff[i+1]), lwd = 4, col = 'blue')

  if(any(is.na(shockoff))){
    shockoff[is.na(shockoff)]<-mean(shockoff, na.rm = TRUE)
  }

  # Check if the sciplot package is installed, install it if necessary
  if (!requireNamespace("sciplot", quietly = TRUE)) {
    install.packages("sciplot")
  }

  library(sciplot)
  par(xpd = T)

  # Create the label vector for bar graph
  labels <- c(rep(paste('No', event), length(shockoff)),  rep(event, length(shockon)))

  # Calculate the mean for no-shock and shock conditions
  no_shock_mean <- shockoff
  shock_means <- shockon

  # Combine the means and calculate the relative means
  means <- c(no_shock_mean, shock_means)
  relative_means <- means / mean(no_shock_mean, na.rm=TRUE) * 100

  if(mean(shock_means, na.rm=TRUE)>mean(no_shock_mean, na.rm=TRUE)){
    ymax=120
  }else{
    ymax=102
  }

  bargraph.CI(labels, relative_means, ylab = "Grand mean (%)", las = 1, ylim = c(0, ymax), xlab = '', col = c('white', col))
  par(xpd = F)

  # Create the list object with average velocity during shock and non-shock conditions
  output_list <- list(
    during_shock = shockon,
    non_shock = shockoff,
    perc.red = 100 - mean(shockon)/mean(shockoff) * 100
  )
  print(t.test(shockon, shockoff))
  return(output_list)

}
