document.addEventListener('DOMContentLoaded', function() {
    const classifyBtn = document.getElementById('classifyBtn');
    const clearBtn = document.getElementById('clearBtn');
    const sampleBtn = document.getElementById('sampleBtn');
    const textInput = document.getElementById('textInput');
    const charCount = document.getElementById('charCount');
    const results = document.getElementById('results');
    const category = document.getElementById('category');
    const confidence = document.getElementById('confidence');
    const levelAnalysis = document.getElementById('levelAnalysis');
    const levelChart = document.getElementById('levelChart');
    const levelSummaryContent = document.getElementById('levelSummaryContent');
    const warningsContainer = document.getElementById('warningsContainer');
    const warningsList = document.getElementById('warningsList');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');
    const errorDetails = document.getElementById('errorDetails');


    // 示例文本
    const sampleTexts = [
        '人文之元，肇自太极，幽赞神明，《易》象惟先。庖牺画其始，仲尼翼其终。而《乾》《坤》两位，独制《文言》。言之文也，天地之心哉！若乃《河图》孕乎八卦，《洛书》韫乎九畴，玉版金镂之实，丹文绿牒之华，谁其尸之？亦神理而已。自鸟迹代绳，文字始炳，炎皞遗事，纪在《三坟》，而年世渺邈，声采靡追。唐虞文章，则焕乎始盛。元首载歌，既发吟咏之志；益稷陈谟，亦垂敷奏之风。夏后氏兴，业峻鸿绩，九序惟歌，勋德弥缛。逮及商周，文胜其质，《雅》《颂》所被，英华日新。文王患忧，繇辞炳曜，符采复隐，精义坚深。重以公旦多材，振其徽烈，制诗缉颂，斧藻群言。至若夫子继圣，独秀前哲，镕钧六经，必金声而玉振；雕琢情性，组织辞令，木铎启而千里应，席珍流而万世响，写天地之辉光，晓生民之耳目矣。爰自风姓，暨于孔氏，玄圣创典，素王述训，莫不原道心以敷章，研神理而设教，取象乎《河》《洛》，问数乎蓍龟，观天文以极变，察人文以成化；然后能经纬区宇，弥纶彝宪，发挥事业，彪炳辞义。故知：道沿圣以垂文，圣因文而明道，旁通而无滞，日用而不匮。《易》曰∶“鼓天下之动者存乎辞。”辞之所以能鼓天下者，乃道之文也。赞曰：道心惟微，神理设教。光采元圣，炳耀仁孝。龙图献体，龟书呈貌。天文斯观，民胥以效。',
        '晋侯、秦伯围郑，以其无礼于晋，且贰于楚也。晋军函陵，秦军氾南。佚之狐言于郑伯曰：“国危矣，若使烛之武见秦君，师必退。”公从之。辞曰：“臣之壮也，犹不如人；今老矣，无能为也已。”公曰：“吾不能早用子，今急而求子，是寡人之过也。然郑亡，子亦有不利焉！”许之。夜缒而出，见秦伯，曰：“秦、晋围郑，郑既知亡矣。若亡郑而有益于君，敢以烦执事。越国以鄙远，君知其难也，焉用亡郑以陪邻？邻之厚，君之薄也。若舍郑以为东道主，行李之往来，共其乏困，君亦无所害。且君尝为晋君赐矣，许君焦、瑕，朝济而夕设版焉，君之所知也。夫晋，何厌之有？既东封郑，又欲肆其西封，若不阙秦，将焉取之？阙秦以利晋，唯君图之。”秦伯说，与郑人盟。使杞子、逢孙、杨孙戍之，乃还。子犯请击之。公曰：“不可。微夫人之力不及此。因人之力而敝之，不仁；失其所与，不知；以乱易整，不武。吾其还也。”亦去之。',
        '陈康肃公尧咨善射，当世无双 ，公亦以此自矜。尝射于家圃，有卖油翁释担而立，睨之，久而不去。见其发矢十中八九，但微颔之。康肃问曰：“汝亦知射乎？吾射不亦精乎？”翁曰：“无他， 但手熟尔。”康肃忿然曰：“尔安敢轻吾射?”翁曰：“以我酌油知之。”乃取一葫芦置于地，以钱覆其口，徐以杓酌油沥之，自钱孔入，而钱不湿。因曰：“我亦无他， 惟手熟尔。”康肃笑而遣之。'
    ];

    // 字符计数功能
    textInput.addEventListener('input', function() {
        const count = this.value.length;
        charCount.textContent = count;
        
        if (count > 3000) {
            charCount.classList.add('exceeded');
            classifyBtn.disabled = true;
        } else {
            charCount.classList.remove('exceeded');
            classifyBtn.disabled = false;
        }
    });

    
    // 分析按钮点击事件
    classifyBtn.addEventListener('click', function() {
        analyzeText();
    });
    
    // 清空按钮点击事件
    clearBtn.addEventListener('click', function() {
        textInput.value = '';
        charCount.textContent = '0';
        results.style.display = 'none';
        levelAnalysis.style.display = 'none';
        error.style.display = 'none';
        warningsContainer.style.display = 'none';
        charCount.classList.remove('exceeded');
        classifyBtn.disabled = false;
    });
    
    // 示例文本按钮点击事件
    sampleBtn.addEventListener('click', function() {
        const randomIndex = Math.floor(Math.random() * sampleTexts.length);
        textInput.value = sampleTexts[randomIndex];
        charCount.textContent = textInput.value.length;
    });
    
    // 分析文本函数
    function analyzeText() {
        const text = textInput.value.trim();
        
        if (!text) {
            showError('请输入文本进行分析');
            return;
        }
        
        if (text.length > 3000) {
            showError('文本长度超过3000字，请减少内容');
            return;
        }
        
        // 显示加载状态
        loading.style.display = 'block';
        results.style.display = 'none';
        levelAnalysis.style.display = 'none';
        error.style.display = 'none';
        warningsContainer.style.display = 'none';
        
        // 准备请求数据
        const formData = new FormData();
        formData.append('text', text);
        
        // 发送请求
        fetch('/classify', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loading.style.display = 'none';
            
            if (data.error) {
                // 处理异常检测错误
                if (data.error_type === 'text_quality') {
                    showQualityError(data);
                } else {
                    showError(data.error);
                }
                return;
            }
            
            // 显示分类结果
            displayResults(data);
            
            // 显示层次分析
            if (data.level_analysis && data.level_chart) {
                displayLevelAnalysis(data.level_analysis, data.level_chart);
                levelAnalysis.style.display = 'block';
            }
            
            // 显示警告信息（如果有）
            if (data.warnings && data.warnings.length > 0) {
                showWarnings(data.warnings, data.warning_suggestions);
            }
            
            results.style.display = 'block';
            results.scrollIntoView({ behavior: 'smooth', block: 'start' });
        })
        .catch(err => {
            loading.style.display = 'none';
            showError('请求失败: ' + err.message);
        });
    }
    
    // 显示质量检测错误
    function showQualityError(data) {
        error.style.display = 'block';
        errorMessage.textContent = data.error;
        
        let detailsHtml = '';
        
        if (data.error_details && data.error_details.length > 0) {
            detailsHtml += '<h4>具体问题：</h4><ul>';
            data.error_details.forEach(detail => {
                detailsHtml += `<li>${detail}</li>`;
            });
            detailsHtml += '</ul>';
        }
        
        if (data.suggestions && data.suggestions.length > 0) {
            detailsHtml += '<h4>改进建议：</h4><ul>';
            data.suggestions.forEach(suggestion => {
                detailsHtml += `<li>${suggestion}</li>`;
            });
            detailsHtml += '</ul>';
        }
        
        errorDetails.innerHTML = detailsHtml;
    }
    
    // 显示警告信息
    function showWarnings(warnings, suggestions) {
        if (!warnings || warnings.length === 0) return;
        
        let warningsHtml = '';
        warnings.forEach((warning, index) => {
            const suggestion = suggestions && suggestions[index] ? suggestions[index] : '';
            warningsHtml += `
                <div class="warning-item">
                    <div class="warning-message">${warning}</div>
                    ${suggestion ? `<div class="warning-suggestion">建议：${suggestion}</div>` : ''}
                </div>
            `;
        });
        
        warningsList.innerHTML = warningsHtml;
        warningsContainer.style.display = 'block';
    }
    
    // 显示正常分析结果
    function displayResults(data) {
        console.log("Complete response data:", data);
        
        // 显示分类结果
        category.textContent = data.category;
        
        // 显示概率分布（2位小数）
        let confidenceHtml = '';
        if (data.probabilities && Object.keys(data.probabilities).length > 0) {
            confidenceHtml = '<p>各类别概率：</p><ul style="list-style-position:inside">';
            Object.entries(data.probabilities).forEach(([cat, prob]) => {
                const percentage = (prob * 100).toFixed(2);  // 2位小数
                confidenceHtml += `<li>${cat}: <strong>${percentage}%</strong></li>`;
            });
            confidenceHtml += '</ul>';
        }
        confidence.innerHTML = confidenceHtml;
        
    }
    
    // 显示层次分析
    function displayLevelAnalysis(levelAnalysisData, levelChartData, levelReportData) {
        console.log("Level analysis data:", levelAnalysisData);
        
        // 显示层次分析图表
        if (levelChartData) {
            levelChart.src = 'data:image/png;base64,' + levelChartData;
            levelChart.style.display = 'block';
        }
        
        // 显示层次总结
        if (levelAnalysisData) {
            displayLevelSummary(levelAnalysisData);
        }
    }
    
    // 显示层次总结
    function displayLevelSummary(levelData) {
        let summaryHtml = '';
        summaryHtml += '<div class="level-summary-grid">';
        
        Object.entries(levelData).forEach(([levelKey, result]) => {
            const score = result.avg_score;
            let gradeClass = '';
            let grade = '';
            
            if (score >= 8) {
                grade = "很高";
                gradeClass = "grade-very-high";
            } else if (score >= 6) {
                grade = "较高";
                gradeClass = "grade-high";
            } else if (score >= 4) {
                grade = "中等";
                gradeClass = "grade-medium";
            } else {
                grade = "较低";
                gradeClass = "grade-low";
            }
            
            summaryHtml += `
                <div class="level-summary-item">
                    <h5>${result.name}</h5>
                    <div class="score-display">
                        <span class="score-value">${score.toFixed(2)}</span>
                        <span class="score-grade ${gradeClass}">${grade}</span>
                    </div>
                    <p class="description">${result.description}</p>
                    <div class="feature-count">包含 ${Object.keys(result.feature_scores).length} 个特征</div>
                </div>
            `;
        });
        
        summaryHtml += '</div>';
        levelSummaryContent.innerHTML = summaryHtml;
    }
    
    
    // 显示错误信息
    function showError(message) {
        error.style.display = 'block';
        errorMessage.textContent = message;
        errorDetails.innerHTML = '';
    }
});