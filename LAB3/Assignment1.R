# -------------------------------------------
# Title: Kernel methods
# Author: Erik Hjalmarsson
# File: Assignment1.R
# Date: 2023-12-06
# Last modified: 2023-12-13
# -------------------------------------------

# Load necessary libraries
library(geosphere)

#--------------------------------------------
# Code
#--------------------------------------------
set.seed(1234567890)

# Import the dataset
stations <- read.csv("stations.csv", fileEncoding = "latin1")
temps <- read.csv("temps50k.csv")

st <- merge(stations, temps, by = "station_number")

h_distance <- 10000# These three values are up to the students
h_date     <- 14
h_time     <- 6

# Central Stockholm Metropolitan
a <- 59.329323 # Latitude
b <- 18.068581 # Longitude

date <- "2013-02-01" # The date to predict (up to the students)
times <- c(
  "04:00:00",
  "06:00:00",
  "08:00:00",
  "10:00:00",
  "12:00:00",
  "14:00:00",
  "16:00:00",
  "18:00:00",
  "20:00:00",
  "22:00:00",
  "24:00:00"
)

temp_kernelprod <- vector(length = length(times))

temp_kernelsum <- vector(length = length(times))

#--------------------------------------------
# Students Code
#--------------------------------------------

location <- c(a, b)

# Remove all dates after date variable including date
filtered_st <- st[st$date < date, ]

# Gaussian kernel function
gaussian_kernel <- function(x, h) {
  return(exp(-((x / h) ^ 2)))
}

#--------------------------------------------
# Kernel for distance
#--------------------------------------------

dist <- distHaversine(filtered_st[, c("latitude", "longitude")],
                      location)
kv_dist <- gaussian_kernel(dist, h_distance)

##--------- Kernel value plot ---------##
plot_length <- 20
kv_dist_plot <- vector(length = plot_length +1)
distance <- vector(length = plot_length +1)

for (i in seq(0, plot_length)){
  kv_dist_plot[i+1] <- gaussian_kernel(i*1000, h_distance)
  distance[i+1] <- i
  
}

plot.default(y = kv_dist_plot,
             x = distance,
             ylab = "Kernel Value",
             xlab = "Distance to location km",
             ylim = c(0, 1),
             xaxt = "n",
             type = "b",
             main = "Decrease in kernel value with h_distance = 10000",
             col = "blue",
             pch = 6)

axis(1,
     at = distance,
     labels = distance,
     las = 2)

#--------------------------------------------
# Kernel for day difference
#--------------------------------------------

day_diff <- difftime(date, filtered_st$date, units = "days")

kv_day <- gaussian_kernel((as.numeric(day_diff)), h_date)

#--------------------------------------------
# Kernel for time difference
#--------------------------------------------

time_format <- strptime(times, format = "%H:%M:%S")
filtered_st_tf <- strptime(filtered_st$time, format = "%H:%M:%S")

for (i in seq(times)) {
  hour <- abs(as.numeric(difftime(time_format[i],
                                  filtered_st_tf,
                                  units = "hours")))

  # Kernel Value for time.
  kv_time <- gaussian_kernel(hour, h_time)

  # Sum of all kernel values
  k_sum <- kv_day + kv_dist + kv_time

  # Product of all kernel values
  k_prod <- kv_day * kv_dist * kv_time

  temp_kernelsum[i] <-
    sum(k_sum %*% filtered_st$air_temperature) / sum(k_sum)
  temp_kernelprod[i] <-
    sum(k_prod %*% filtered_st$air_temperature) / sum(k_prod)
}

# Convert times to character format for better formatting on the x-axis
times_labels <- format(times, format = "%H:%M:%S")

# Create a scatter plot
plot.default(
             y = temp_kernelsum,
             x = as.factor(times),
             pch = 1,
             type = "b",
             ylim = c(min(temp_kernelsum, temp_kernelprod)-1,
                      max(temp_kernelsum, temp_kernelprod)+2),
             bg = "red",
             xaxt = "n", # No x-axis yet
             xlab = "",
             ylab = "Temperature") # Y-axis label

points(temp_kernelprod,
       pch = 25,
       type = "b",
       bg = "blue")

# Add x-axis with custom labels
axis(1,
     at = as.factor(times),
     labels = times_labels,
     las = 2)

# Add legend
legend("topright",
       legend = c("Kernel Sum", "Kernel Product"),
       fill = c("white", "blue"),
       bty = "n")

# Add a title
title(main = paste("Temperature", date))

# Add grid lines
grid()
