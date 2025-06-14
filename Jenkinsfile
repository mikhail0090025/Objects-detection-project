pipeline {
    agent any
    stages { 
        stage('Start message') {
            steps {
                echo 'Start of the pipeline'
            }
        }
        stage('Checkout') {
            steps {
                git url: 'https://github.com/mikhail0090025/Objects-detection-project', branch: 'main'
            }
        }
        stage('Start project & Save Logs') {
            steps {
                sh '''
                    python neural_net.py > neural_net.log
                '''
                archiveArtifacts artifacts: 'neural_net.log', allowEmptyArchive: true
            }
        }
    }
    post {
        always {
            echo 'Pipeline finished!'
        }
        success {
            echo 'Test was success!'
        }
        failure {
            echo 'Test was NOT success! Check logs.'
        }
    }
}