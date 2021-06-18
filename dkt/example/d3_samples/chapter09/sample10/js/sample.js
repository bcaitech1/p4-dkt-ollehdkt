// JSON 데이터 불러오기
d3.json("mydata.json", function(error, data){
	var dataSet = [ ];	// 데이터셋
	// SVG 요소의 넓이와 높이를 구함
	var svgEle = document.getElementById("myGraph");
	var svgWidth = window.getComputedStyle(svgEle, null).getPropertyValue("width");
	var svgHeight = window.getComputedStyle(svgEle, null).getPropertyValue("height");
	svgWidth = parseFloat(svgWidth) - 60;	// 값에는 단위가 있으므로 단위를 삭제
	svgHeight = parseFloat(svgHeight) - 60;	// 값에는 단위가 있으므로 단위를 삭제
	var offsetX = 30;	// 가로 오프셋
	var offsetY = 20;	// 세로 오프셋
	var scale = 2.0;	// 2배 크기로 그리기
	var rangeYear = 10;	// 10년분을 표시
	// 최댓값과 최솟값 연도를 구함
	var year = d3.extent(data, function(d){
		return d.year;
	});
	var startYear = year[0];	// 최초 연도
	var currentYear = 2000;	// 최초의 표시 기준 연도
	var margin = svgWidth /(rangeYear - 1);	// 꺾은선 그래프의 간격 계산
	// 처음 그래프 표시
	pickupData(data, currentYear-startYear);	// 표시 범위의 데이터를 입력
	drawGraph(dataSet, "item1", "itemA", "linear");	// item1의 데이터
	drawGraph(dataSet, "item2", "itemB", "linear");	// item2의 데이터
	drawGraph(dataSet, "item3", "itemC", "linear");	// item3의 데이터
	drawScale();	// 눈금 표시
	// 꺾은선 그래프를 표시하는 함수
	function drawGraph(dataSet, itemName, cssClassName, type){
		// 꺾은선 그래프의 좌표를 계산하는 메서드
		var line = d3.svg.line()	// svg의 선
		  .x(function(d, i){
				return offsetX + i * margin;	// X 좌표는 표시 순서×간격
			})
		  .y(function(d, i){
				return svgHeight - (d[itemName] * scale) - offsetY;	// 데이터로부터 Y 좌표 빼기
			})
		  .interpolate(type)	// 꺾은선 그래프의 모양 설정
		// 꺾은선 그래프 그리기
		var lineElements = d3.select("#myGraph")
		  .append("path")	// 데이터 수만큼 path 요소가 추가됨
		  .attr("class", "line "+cssClassName)	// CSS 클래스 지정
		  .attr("d", line(dataSet))	//연속선 지정
	}
	// 그래프의 눈금을 표시하는 함수
	function drawScale(){
		// 눈금을 표시하기 위해 D3 스케일 설정
		var yScale = d3.scale.linear()  // 스케일 설정
		  .domain([0, 100])   // 원래 크기
		  .range([scale*100, 0]) // 실제 표시 크기
		// 눈금 표시
		d3.select("#myGraph")	// SVG 요소를 지정
			  .append("g")	// g 요소 추가. 이것이 눈금을 표시하는 요소가 됨
			  .attr("class", "axis")	// CSS 클래스 지정
			  .attr("transform", "translate("+offsetX+", "+((100-(scale-1)*100)+offsetY)+")")	// ●변경
			  .call(
					d3.svg.axis()
				  .scale(yScale)  //스케일 적용
				  .orient("left") //눈금 표시 위치를 왼쪽으로 지정
				)
		// 가로 눈금을 표시하기 위해 D3 스케일 설정
		var xScale = d3.time.scale()  // 스케일 설정
		  .domain([new Date(currentYear+"/1/1"), new Date((currentYear + rangeYear - 1)+"/1/1")])	// 표시 범위의 연도를 지정
		  .range([0, svgWidth]) // 표시 크기
		// 가로 눈금 표시
		d3.select("#myGraph")	// SVG 요소를 지정
			  .append("g")	// g 요소 추가. 이것이 눈금을 표시하는 요소가 됨
			  .attr("class", "axis")	// CSS 클래스 지정
			  .attr("transform", "translate("+offsetX+", "+(svgHeight - offsetY)+")")
			  .call(
					d3.svg.axis()
				  .scale(xScale)  //스케일 적용
				  .orient("bottom") //눈금 표시 위치를 왼쪽으로 지정
				  .ticks(10)	// 1년마다 표시
				  .tickFormat(function(d, i){
						var fmtFunc = d3.time.format("%Y년%m월");	// 변환 함수
						return fmtFunc(d);	// 변환한 결과를 반환
					})
				)
			  .selectAll("text")	// 눈금 문자를 대상으로 처리
			  .attr("transform", "rotate(90)")	// 90도 회전
			  .attr("dx", "0.7em")	// 위치 조정
			  .attr("dy", "-0.4em")	// 위치 조정
			  .style("text-anchor", "start")	// 표시 위치 지정
	}
	// JSON 데이터로부터 표시할 범위의 데이터셋을 추출하고 SVG 요소 안을 삭제
	function pickupData(data, start){
		dataSet = [ ];	// 데이터셋 삭제
		for(var i=0; i<rangeYear; i++){	// 표시할 범위의 데이터를 입력
			dataSet[i] = data[start + i];
		}
		d3.select("#myGraph").selectAll("*").remove();	// SVG 요소 안 삭제
	}
	// <이전> 버튼에 이벤트를 할당
	d3.select("#prev").on("click", function(){
		if (currentYear > year[0]){	// 최솟값(연)보다 클 때는 연도를 1 감소함
			currentYear = currentYear - 1;
		}
		// 그래프 표시
		pickupData(data, currentYear-startYear);	// 표시 범위의 데이터를 입력
		drawGraph(dataSet, "item1", "itemA", "linear");
		drawGraph(dataSet, "item2", "itemB", "linear");
		drawGraph(dataSet, "item3", "itemC", "linear");
		drawScale();
	})
	// <다음> 버튼에 이벤트를 할당
	d3.select("#next").on("click", function(){
		if (currentYear <= year[1]-rangeYear){	// 최댓값(연)+범위보다 작다면 연도를 1 증가함
			currentYear = currentYear + 1;
		}
		// 그래프 표시
		pickupData(data, currentYear-startYear);	// 표시 범위의 데이터를 입력
		drawGraph(dataSet, "item1", "itemA", "linear");
		drawGraph(dataSet, "item2", "itemB", "linear");
		drawGraph(dataSet, "item3", "itemC", "linear");
		drawScale();
	});
});
