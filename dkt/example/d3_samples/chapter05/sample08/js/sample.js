// 데이터셋이 HTML 파일일 때
d3.html("mydata.html", function(error, docFragment){
	var tr = docFragment.querySelectorAll("table tr");	// table 요소에서 tr 요소를 추출
	var dataSet = [ ];	// 데이터를 저장할 배열을 준비
	for(var i=1; i<tr.length; i++){	// tr 요소의 줄 수-1만큼 반복(1번째 행 헤더는 무시)
		var d = tr[i].querySelectorAll("td")[0].firstChild.nodeValue;	// 데이터 읽어들이기
		dataSet.push(d);	// 상품A의 데이터만 추출
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