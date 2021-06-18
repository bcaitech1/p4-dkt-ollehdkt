// CSV 파일 읽어들이기
d3.csv("mydata.csv", function(error, data){
	var dataSet = [ ];	// 데이터를 저장할 배열 준비
	for(var i=0; i<data.length; i++){	// 데이터 줄 개수만큼 반복
		dataSet.push(data[i]["상품A"]);	// 상품A의 레이블 데이터만 추출
	}
	// 요소의 데이터를 바꿈
	d3.select("#myGraph")
	  .selectAll("rect")	// rect 요소 지정
	  .data(dataSet)	// 데이터를 요소에 연결
	  .enter()	// 데이터 개수만큼 반복
	  .append("rect")	// 데이터의 개수만큼 rect 요소가 추가됨
	  .datum(999)	// 999의 데이터를 모든 요소에 설정
})