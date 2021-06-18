// CSV 파일을 불러와 그래프 그리기
d3.csv("mydata.csv", function(error, data){
	var dataSet = [ ];						// 데이터를 저장할 배열을 준비
	for(var i=0; i<data.length; i++){	// 데이터의 줄 수만큼 반복
		dataSet.push(data[i].item1);	// item1 레이블의 데이터만 추출
	}
	// 데이터를 기준으로 그리기
	d3.select("#myGraph")		// SVG 요소 지정
		.selectAll("rect")			// SVG로 사각형을 표시할 요소를 지정
		.data(dataSet)				// 데이터 설정
		.enter()							// 데이터의 개수에 따라 rect 요소를 생성
		.append("rect")				// SVG 사각형 생성
		.attr("x", 0)					// 가로형 막대그래프이므로 X 좌표를 0으로 함
		.attr("y", function(d,i){	// Y 좌표를 배열의 순서에 따라 계산
			return i * 25;			// 막대그래프의 Y 좌표를 25px 단위로 계산
		})
		.attr("height", "20px")			// 막대그래프의 높이를 20px로 지정
		.attr("width", function(d, i){	// 막대그래프의 넓이를 배열의 내용에 따라 계산
			return d +"px";					// 데이터의 값을 그대로 넓이로 함
		})
});
