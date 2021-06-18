// 데이터셋이 플레인 텍스트 파일일 때
d3.text("mydata.txt", function(error, plainText){
	var data = plainText.split("\x0a");	// \0x0a는 줄바꿈 코드
	var dataSet = [ ];	// 데이터를 저장할 배열을 준비
	var sales = data[0].split("/");	// 처음 1줄을 / 구분자로 나누어 데이터를 분할
	for(var i=1; i<sales.length; i++){
		dataSet.push(sales[i]);	// sales 데이터 추출
	}
	// 그래프 그리기
	d3.select("#myGraph")
	  .selectAll("rect")	// rect 요소 지정
	  .data(dataSet)	// 데이터를 요소에 연결
	  .enter()	// 데이터 개수만큼 반복
	  .append("rect")	// 데이터 개수만큼 rect 요소가 추가됨
	  .attr("class", "bar")	// CSS 클래스를 지정
	  .attr("width", function(d,i){	// 넓이를 지정. 두 번째의 파라미터에 함수를 지정
		return d;	// 데이터 값을 그대로 넓이로 반환
	  })
	  .attr("height", 20)	// 높이를 지정
	  .attr("x", 0)	// X 좌표를 0으로 함
	  .attr("y", function(d, i){	// Y 좌표를 지정함
		return i * 25	// 표시 순서에 25를 곱해 위치를 계산
	  })
})