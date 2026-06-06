file_paths19_20 <- list.files(path = "F:/KasankaCameras", pattern = "*.MP4",
                              recursive = TRUE, full.names = TRUE)
file_paths21 <- list.files(path = "K:/KasankaCameras", pattern = "*.MP4",
                           recursive = TRUE, full.names = TRUE)
file_paths <- c(file_paths19_20, file_paths21)

file_paths <- list.files(path = "E:/", pattern = "*.MP4",
                           recursive = TRUE, full.names = TRUE)
# Initialize an empty data frame
df <- data.frame(Date = character(),
                 Location = character(),
                 FileName = character(),
                 VideoLength = numeric(),
                 FrameSize = character(),
                 FrameRate = numeric(),
                 stringsAsFactors = FALSE)

# Load the necessary libraries
library(pacman)
p_load(tidyverse, dtplyr, lubridate, data.table,
       ggplot2,
       av)

# Loop through each file path
file_path <- file_paths[1]
for (file_path in file_paths) {
  # Split the file path into components
  path_components <- unlist(strsplit(file_path, "/"))

  # Load the video file
  video_info <- av::av_video_info(file_path)  # Extract video length, frame size, and frame rate
  video_length <- video_info$duration
  frame_size <- paste(video_info$video$width, video_info$video$height, sep = "x")
  frame_rate <- video_info$video$framerate

  # Create a temporary data frame with the relevant components
  temp_df <- data.frame(Date = path_components[3],
                        Location = path_components[4],
                        FileName = path_components[5],
                        VideoLength = video_length,
                        FrameSize = frame_size,
                        FrameRate = frame_rate,
                        stringsAsFactors = FALSE)

  # Append the temporary data frame to the main data frame
  df <- rbind(df, temp_df)
}

write.csv(df, file = "../../../Dropbox/MPI/Eidolon/Data/Zambia/Kasanka/video_files22.csv")

df[which(df$FrameSize != "2704x1520"),]
df$FrameRate %>% table
df[which(df$FrameRate == 25),]


k <- readxl::read_xlsx("../../../Dropbox/MPI/Eidolon/Data/Zambia/Kasanka/video_files_checked_master.xlsx")
k$`Trees in frame` %>% table() %>% sum
