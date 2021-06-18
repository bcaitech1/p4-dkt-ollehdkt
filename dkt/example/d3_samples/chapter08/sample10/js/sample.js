// 처음에는 2008년 데이터를 표시해둠
drawPie("mydata2008.csv");
// 선택 메뉴가 선택되었을 때의 처리
d3.select("#year").on("change", function(){
	d3.select("#myGraph").selectAll("*").remove();	// svg 안의 모든 요소를 삭제
	drawPie("mydata"+this.value+".csv", this.value);	// 지정된 연도의 데이터를 불러와 원 그래프 표시
});
function drawPie(filename){
// 데이터셋은 CSV 파일
	d3.csv(filename, function(error, data){
		var dataSet = [ ];	// 데이터를 저장할 배열 변수
		for(var i in data[0]){	// 최초 데이터를 처리
			dataSet.push(data[0][i]);	// 가로 한 줄 모두를 한꺼번에 입력
		}
		// SVG 요소의 넓이와 높이를 구함
		var svgEle = document.getElementById("myGraph");
		var svgWidth = window.getComputedStyle(svgEle, null).getPropertyValue("width");
		var svgHeight = window.getComputedStyle(svgEle, null).getPropertyValue("height");
		svgWidth = parseFloat(svgWidth);	// 값에는 단위가 붙어 있으므로 단위를 삭제
		svgHeight = parseFloat(svgHeight);	// 값에는 단위가 붙어 있으므로 단위를 삭제
		// 원 그래프의 좌표값을 계산하는 메서드
		var pie = d3.layout.pie()	// 원 그래프 레이아웃
		// 원 그래프의 안쪽 반지름, 바깥쪽 반지름 설정
		var arc = d3.svg.arc().innerRadius(30).outerRadius(100)
		// 원 그래프 그리기
		var pieElements = d3.select("#myGraph")
		  .selectAll("g")	// g 요소 지정
		  .data(pie(dataSet))	// 데이터를 요소에 연결
		  .enter()
		  .append("g")	// 무게 중심 계산을 위하 그룹화하기
		  .attr("transform", "translate("+svgWidth/2+", "+svgHeight/2+")")    // 원 그래프의 중심으로 함
		// 데이터 추가
		pieElements	// 데이터 수만큼 반복
		  .append("path")	// 데이터의 수만큼 path 요소가 추가됨
		  .attr("class", "pie")	// CSS 클래스 설정
		  .style("fill", function(d, i){
				return ["#ff3344", "#ff7328", "#d3d4d5", "#dfd"][i];	// 통신사의 색을 반환
			})
		  .transition()
		  .duration(200)
		  .delay(function(d,i){   // 그릴 원 그래프의 시간을 어긋나게 표시
				return i*200;
			})
		  .ease("linear")	// 직선적인 움직임으로 변경
		  .attrTween("d", function(d, i){	// 보간 처리
				var interpolate = d3.interpolate(
					{ startAngle : d.startAngle, endAngle : d.startAngle }, // 각 부분의 시작 각도
					{ startAngle : d.startAngle, endAngle : d.endAngle }    // 각 부분의 종료 각도
       			 );
				return function(t){
					return arc(interpolate(t)); // 시간에 따라 처리
				}
			})
		// 합계와 문자 표시
		var textElements = d3.select("#myGraph")
		  .append("text")	// text 요소 추가
		  .attr("class", "total")	// CSS 클래스 설정
		  .attr("transform", "translate("+svgWidth/2+", "+(svgHeight/2+5)+")")    // 가운데에 표시
		  .text("점유율")	// 문자 표시
		// 숫자를 부채꼴의 가운데 표시
		pieElements
		  .append("text")	// 데이터의 수만큼 text 요소가 추가됨
		  .attr("class", "pieNum")	// CSS 클래스 설정
		  .attr("transform", function(d, i){
				return "translate("+arc.centroid(d)+")";    // 부채꼴의 중심으로 함
			})
		  .text(function(d, i){
				return d.value;	// 값 표시
			})
	});
}
