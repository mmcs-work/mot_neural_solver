# #!/bin/sh
mkdir testset
cd testset
wget -O cont_det.zip "https://motchallenge.net/download_results.php?shakey=e9aa2700c84ee481cf512068353ef2df3ae885ce&name=YOLOV3M2CTMC&chl=19"

for zip in *.zip
do
  dirname=`echo $zip | sed 's/\.zip$//'`
  if mkdir "$dirname"
  then
    if cd "$dirname"
    then
      unzip ../"$zip"
      cd ..
      # rm -f $zip # Uncomment to delete the original zip file
    else
      echo "Could not unpack $zip - cd failed"
    fi
  else
    echo "Could not unpack $zip - mkdir failed"
  fi
done
