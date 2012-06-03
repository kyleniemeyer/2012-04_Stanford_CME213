for dir in ~/benchmarksuite/*/
do
  echo ${dir}
  cd ${dir}
  fp ${dir}a.txt ${dir}x.txt
done

