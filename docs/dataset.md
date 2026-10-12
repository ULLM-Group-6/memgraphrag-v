text_manifest.jsonl Actual caption text is stored in this file eg {
  "doc_id": "Center (gridiron football) - Wikipedia",
  "passage_id": "txt_d5df4c700dba11ecb1e81171463288e9_9",
  "text": "The center is the innermost lineman ...",
  "source_url": "https://en.wikipedia.org/wiki/Center_(American_football)"
}


image_manifest.jsonl to identify image nodes and their metadata. eg {
  "doc_id": "Bandol rose",
  "passage_id": "cap_30388563", ID of the associated caption passage.
  "text": "Caption text for the image...",
  "source_url": "https://commons.wikimedia.org/..."
}


images_to_download.jsonl to download the corpus images, then process those local image files.


build_qa.json to inspect the graph-building questions and their evidence.


test_qa.json only for evaluation

Linux command to download the main data
gdown --folder "https://drive.google.com/drive/folders/1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ" --remaining-ok  50 chunks only

For remaining chunks, 2 files were scp from local :)
while read -r id name; do
    if [ -s "$DEST/$name" ]; then
        echo "Skipping existing: $name"
    else
        echo "Downloading missing: $name"
        gdown "https://drive.google.com/uc?id=$id" -O "$DEST/$name" || {
            echo "Failed: $name"
            break
        }
    fi
done <<'EOF'
1bcINVUPzlZRs_4ZImisZXmQOZFkQCNem imgs.7z.001
1dRqJ8f1G4WcMd0f2G3grQ19x1yh1ds1R imgs.7z.002
1W5PvSD235zoowI5fwel0wrsGeLBZchlD imgs.7z.003
1IQ1T-GS80ENqkF6NHR_n3AMMssBeLH9Q imgs.7z.004
1eXpaG3Hcef0XMwPybCBPD1XMIYupdlef imgs.7z.005
1yLBG2efdHY2iQEOOKWVcMcog8LIflofh imgs.7z.006
14q7OG5KW1NwnOv9LEh16y0By7sQKdvQ_ imgs.7z.007
1Nh58uHq292bnulwq1VC0LErtKxvQK-Jg imgs.7z.008
1hz2KxfX6h_Xxd7N3jvmt9vwzGJ_CdO6W imgs.7z.009
186KMJK4tfmn6ZIyNZRjKEXrCIF_m3ZVV imgs.7z.010
1yqeLyPbs7pfOJOhECC2hc5WWlfCbmh5w imgs.7z.011
1eGFCsNI302Mh2UzA1u3Sk4a0oYydfEIu imgs.7z.012
1UScno01nENyfFI7D9VNVANqkfD1Yt7Vs imgs.7z.013
1Bl0FSTr33Kk8qxVEaIFdxY2tHNt40pri imgs.7z.014
1nElqrscQvUj1ngKqL7enq9pXziwuU9oC imgs.7z.015
1QGPdjrCj7-_6WXQ3vWp32dK02_IGoGmT imgs.7z.016
1D4fKQkyRY3S_gEhlExOT87m46ctP00hu imgs.7z.017
1BNH-OVZITjgjvT_uwy4TV2lWtOd-V_dE imgs.7z.018
1dwhhV3vnkblBcBd84rVS5WLnGHX2fYjM imgs.7z.019
1amtO56w_HwvpnSYXZF_Ygv1tOyP0SNDD imgs.7z.020
1udUPb97NYy2w7_RTs8exoyKt8QBkwT_d imgs.7z.021
1ij5Ib7ILJgK0Dvo_FS6Mo53zSe2LwEF6 imgs.7z.022
1R4hvoWakNDcPXuIVN4loDquX_6GkI31u imgs.7z.023
1sh5BD_7h7EOe0RQp0trEBc9rp1_SgA4q imgs.7z.024
1fzrA9wmaCrQPnzfNrCWTIw9dvcBW46Ew imgs.7z.025
1CPAql6cDl4Q9aZrlcU5_XyMWOR3WC_ZD imgs.7z.026
18o4W7yebd_K9bZeA0Y2-XMkpVPauVJWs imgs.7z.027
1ImJIpoeWU-uEdzFoqj3XscgSzi6ehxDm imgs.7z.028
1z9SjVoxtUJAyi4n9z4ruY0HrwxP-WvrY imgs.7z.029
1Xff0miSsyTWOvRyFg8IxcvF0aMiKZbKd imgs.7z.030
1WmSvzDyCzbkujEG5b37vFKIMZexbfiAF imgs.7z.031
1kELukL5pFYJ_0rD3jrPuad9SFyGq_YWx imgs.7z.032
1ncIXTy8Ow4UTStqqq7w6bL9OKMShbxQA imgs.7z.033
1Taa8OC218LroXukGYOwVqXr1PVnJqnCd imgs.7z.034
1_z8NJGKm7Qgp3voGnoOQelLW5QBnqRv8 imgs.7z.035
1W-ptn_8Fc65Xk1KJs_iHhkSWigUCfv50 imgs.7z.036
1KkeeBiVhzjycYaKxh1iBN4KpUFG9p9LB imgs.7z.037
1qlD5c9TDVUwe7rvBNVz4f9CAQcvOcfME imgs.7z.038
1kg2V8D39oHix1DWjU6eKPxzjm9iNN8u2 imgs.7z.039
1Jd87Tb0eDcVb7dNv0MU313f038WRodZg imgs.7z.040
13Fnr1cvENrwXLmQoBIuLjetUAej6PISb imgs.7z.041
15mDTHOSR89IsjvjMz7SEaJfEYj8wT64Y imgs.7z.042
18BRzMTR3fZRCCROKII-5KQhQodR0ccA8 imgs.7z.043
1zpP-OkK3LVG6plw6iljqLd7_wFyFJGtZ imgs.7z.044
1lgIH__QOyZI06x5Jvbk9_lV-cWkh5Jtn imgs.7z.045
1PmCM9GjN7HeQ40P4Xr8H-h_-_1W2wEmT imgs.7z.046
1JdHDVdbHKQM5tDpFPG0kWf6ye2JlhFYn imgs.7z.047
1R4NDmWXT0y8LBNSJfYqyR_PPGJrpjc8K imgs.7z.048
1192XDyDZbhwuKi8XAR4GCShicqxtqrK3 imgs.7z.049
18bq8Z3O6XNPRuAcxDHhFcYRLwCyodogr imgs.7z.050
EOF
