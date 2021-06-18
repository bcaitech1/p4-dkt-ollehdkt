(function(){
	// SVG 요소의 넓이와 높이를 구함
	var svgEle = document.getElementById("myGraph");
	var svgWidth = window.getComputedStyle(svgEle, null).getPropertyValue("width");
	var svgHeight = window.getComputedStyle(svgEle, null).getPropertyValue("height");
	svgWidth = parseFloat(svgWidth);	// 값에 단위가 포함되었으므로 단위를 삭제
	svgHeight = parseFloat(svgHeight);	// 값에 단위가 포함되었으므로 단위를 삭제
	var blockSize = 20;	// 블록 크기
	var heatMap;	// 히트맵 객체를 저장할 변수
	var color;	// 히트맵의 색을 처리할 함수를 넣을 변수
	var maxValue;	// 데이터의 최댓값
	var dataSet = [ ];	// 데이터셋
	// 데이터 읽어오기
	d3.text("mydata.txt", function(error, plainText){
		var temp = plainText.split(",");	// 반점으로 분할하여 대입
		for(var i=0; i<temp.length; i++){
			dataSet[i] = parseInt(temp[i]);	// 정수로 변환하고 대입
		}
		drawHeatMap();
		// 히트맵을 일정 간격으로 갱신
		setInterval(function(){
			for(var i=0; i<dataSet.length; i++){
				var n = ((Math.random() * 3.5) | 0) - 2;	// 난수 값
				dataSet[i] = dataSet[i] + n;	// 더하기
				if (dataSet[i] < 0){ dataSet[i] = 0; }	// 음수가 되지 않도록 조정
				if (dataSet[i] > maxValue ){ dataSet[i] = maxValue; }	// 최댓값을 넘지 않도록 조정
			}
			heatMap.data(dataSet)
			  .style("fill", function(d, i){	// 색 표시
					return color(d/maxValue);
				})
		}, 1000);
	})
	// 히트맵 표시할 함수
	function drawHeatMap(){
		// 히트맵에 표시할 색을 자동으로 계산
		color = d3.interpolateHsl("blue", "yellow");	// 파란색에서 노란색으로 보간
		maxValue = d3.max(dataSet);	// 최댓값을 구함
		// 히트맵 준비
		heatMap = d3.select("#myGraph")
		  .selectAll("rect")   // rect 요소 지정
		  .data(dataSet)    // 데이터 설정
		// 히트맵 표시
		heatMap.enter()
		  .append("rect")  // rect 요소를 추가
		  .attr("class", "block")	// CSS 클래스 추가
		  .attr("x", function(d, i) { // X 좌표 설정
				return (i % 8) * blockSize;
			}) 
		  .attr("y", function(d, i) { // Y 좌표 설정
				return Math.floor(i/8)*blockSize;
			})
		  .attr("width", function(d, i) {	// 넓이 설정
				return blockSize;
			})
		  .attr("height", function(d, i) {	// 높이 설정
				return blockSize;
			})
		  .style("fill", function(d, i){	// 색 표시
				return color(d/maxValue);
			})
	}
})();

