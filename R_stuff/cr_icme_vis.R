cr_icmes <- read.csv("icme_catalog.csv")

# time is in UTC by cane&richardson list
cr_icmes$disturbance_datetime_ut <-
  as.POSIXct(cr_icmes$disturbance_datetime_ut,
             format="%Y/%m/%d %H%M", tz = "UTC")
cr_icmes$icme_plasma_field_start_ut <-
  as.POSIXct(cr_icmes$icme_plasma_field_start_ut,
             format="%Y/%m/%d %H%M", tz = "UTC")
cr_icmes$icme_plasma_field_end_ut <-
  as.POSIXct(cr_icmes$icme_plasma_field_end_ut,
             format="%Y/%m/%d %H%M", tz = "UTC")

cr_icmes$comp_start_hrs <- as.integer(cr_icmes$comp_start_hrs)
cr_icmes$comp_end_hrs <- as.integer(cr_icmes$comp_end_hrs)

cr_icmes$mc_start_hrs <- as.integer(cr_icmes$mc_start_hrs)
cr_icmes$mc_end_hrs <- as.integer(cr_icmes$mc_end_hrs)
# 1. Get subset
data_subset <- cr_icmes

# 2. Create an empty plot frame
# We set ylim to c(0.5, 1.5) to keep the line centered at y = 1
plot(data_subset$disturbance_datetime_ut, rep(1, nrow(data_subset)),
     type = "n", 
     yaxt = "n", 
     ylim = c(0.8, 1.2), # Tighten vertical space
     xlab = "Time (UTC)", 
     ylab = "",
     main = "ICME Event Timelines (1D)")

# 3. Add the 'Linking Line' (Plasma Field Start to End) at y = 1
segments(x0 = data_subset$icme_plasma_field_start_ut, 
         y0 = 1,
         x1 = data_subset$icme_plasma_field_end_ut, 
         y1 = 1,
         col = "gray", lwd = 2)

# 4. Add the 'Pings' at y = 1
# Black: Disturbance
points(data_subset$disturbance_datetime_ut, rep(1, nrow(data_subset)), 
       pch = "|", col = "black", cex = 2)

# Green: Plasma Start
points(data_subset$icme_plasma_field_start_ut, rep(1, nrow(data_subset)), 
       pch = "|", col = "green3", cex = 2)

# Red: Plasma End
points(data_subset$icme_plasma_field_end_ut, rep(1, nrow(data_subset)), 
       pch = "|", col = "red", cex = 2)

# 5. Add a legend
legend("topright", legend=c("Disturbance", "Plasma Start", "Plasma End"), 
       col=c("black", "green3", "red"), pch="|", bty="n")


# ----------- filter to one solar cycle ---------
read_omni_data <- function(fmt_file, data_file) {
  # 1. Read the format file as text
  fmt_lines <- readLines(fmt_file)
  
  # 2. Extract column names and Fortran formats using Regex
  # This looks for: [Index] [Title] [Format (e.g. F6.1 or I4)]
  # We look for lines that end with a letter and a number (the format)
  fmt_pattern <- "^\\s*\\d+\\s+(.*?)\\s+([IF][\\d.]+)\\s*$"
  matches <- regexec(fmt_pattern, fmt_lines)
  match_list <- regmatches(fmt_lines, matches)
  
  # Filter out lines that didn't match the metadata pattern
  meta_data <- do.call(rbind, lapply(match_list, function(x) {
    if(length(x) == 3) return(c(name = trimws(x[2]), fmt = x[3]))
  }))
  
  titles <- meta_data[, "name"]
  fmt_strings <- meta_data[, "fmt"]
  
  # 3. Convert Fortran formats (e.g., "F6.1", "I4") to numeric widths
  # We just need the number after the letter
  widths <- as.numeric(gsub("[IF]", "", fmt_strings))
  
  # 4. Read the Fixed Width File
  # We skip any header in the .lst file (usually NASA puts a few lines at the top)
  df <- read.fwf(data_file, 
                 widths = widths, 
                 col.names = titles, 
                 header = FALSE,
                 strip.white = TRUE)
  
  # 5. Clean up "Fill Values" (NASA uses 999.9, 99.9, etc. for missing data)
  # Commonly anything > 998 in these fields is a placeholder
  df[df == 99.9 | df == 999.9 | df == 9999 | df == 999999] <- NA
  
  return(df)
}

# may 1996 to nov 2008 is one solar cycle
cutoff1 <- as.POSIXct("1996-05-01", tz = "UTC")
cutoff2 <- as.POSIXct("2008-11-01", tz = "UTC")

# find the index of the FIRST row that is on or after November 2008
start_index <- which(cr_icmes$disturbance_datetime_ut >= cutoff1)[1]
end_index <- which(cr_icmes$disturbance_datetime_ut >= cutoff2)[1]

data_subset <- cr_icmes[1:end_index, ]
