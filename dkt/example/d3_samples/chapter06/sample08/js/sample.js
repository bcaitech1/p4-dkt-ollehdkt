// 버튼을 클릭하면 해당하는 CSV 파일 읽어들이기
d3.selectAll("button").on("click", function(){
	var csvFile = this.getAttribute("data-src");	// data-src 속성을 읽어옴(즉, CSV 파일 이름)
	var barElements;	// 막대그래프의 막대 요소를 저장할 변수
	// CSV 파일을 불러와 그래프 표시
	d3.csv(csvFile, function(error, data){
		var dataSet = [ ];	// 데이터를 저장할 배열 준비
		for(var i=0; i<data.length; i++){	// 데이터 줄 개수만큼 반복
			dataSet.push(data[i]["상품A"]);	// 상품A의 레이블 데이터만 추출
		}
		// 그래프 그리기
		barElements = d3.select("#myGraph")
		  .selectAll("rect")	// rect 요소 지정
		  .data(dataSet)	// 데이터를 요소에 연결
		//데이터가 추가될 때
		barElements.enter()	// 데이터 개수만큼 반복
		  .append("rect")	// 데이터의 개수만큼 rect 요소가 추가됨
		  .attr("class", "bar")	// CSS 클래스를 지정
		  .attr("width", function(d,i){	// 넓이 지정. 2번째 파라미터에 함수 지정
			return d;	// 데이터 값을 그대로 넓이로 반환
		  })
		  .attr("height", 20)	// 높이 지정
		  .attr("x", 0)	//  X 좌표를 0으로 함
		  .attr("y", function(d, i){	// Y 좌표를 지정함
			return i * 25	// 표시 순서에 25를 곱하여 위치를 계산
		  })
		// 데이터가 갱신될 때
		barElements
		  .attr("width", function(d,i){	// 넓이 지정. 2번째 파라미터에 함수 지정
			return d;	// 데이터 값을 그대로 넓이로 반환
		  })
		// 데이터가 삭제될 때
		barElements
		  .exit()	// 삭제 대상 요소 추출
		  .remove()	// 요소 삭제
	})
})
