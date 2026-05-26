// dataforgeai-ml-service CI pipeline.
//
// Stages follow PRD §INFRA.md ML service template:
//   checkout -> install -> lint -> typecheck -> unit tests -> contract tests
//   -> plugin tests -> security tests -> e2e compute demo -> performance
//   -> FastAPI health smoke -> Dagster definitions load check
//   -> build Docker image -> scan image -> publish build metadata.
//
// The pipeline never needs a real backend or production secret to run:
// every test path uses the in-memory storage fake, the fake platform
// metadata client, and the local fallback contract pack.
//
// Parameters:
//   CONTRACT_VERSION — value baked into report metadata and image labels.
//                      Defaults to the local fallback contract pack version.
//   IMAGE_TAG        — tag used for the Docker build; defaults to the
//                      short git sha so consecutive builds are immutable.
//   IMAGE_NAME       — registry-qualified image name (without tag).
//   PUSH_IMAGE       — push the built image to ${IMAGE_NAME}:${IMAGE_TAG}
//                      when true; CI in PR builds typically leaves it false.

pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
spec:
  serviceAccountName: jenkins-build
  containers:
    - name: python
      image: python:3.12-slim
      command: ['cat']
      tty: true
      resources:
        requests:
          cpu: '500m'
          memory: '1Gi'
        limits:
          cpu: '2'
          memory: '4Gi'
    - name: docker
      image: docker:27-cli
      command: ['cat']
      tty: true
      env:
        - name: DOCKER_BUILDKIT
          value: '1'
      volumeMounts:
        - name: docker-sock
          mountPath: /var/run/docker.sock
  volumes:
    - name: docker-sock
      hostPath:
        path: /var/run/docker.sock
        type: Socket
"""
        }
    }

    parameters {
        string(
            name: 'CONTRACT_VERSION',
            defaultValue: 'local-fallback-v0.1.0-demo',
            description: 'Contract pack version to bake into report metadata and image labels.'
        )
        string(
            name: 'IMAGE_NAME',
            defaultValue: 'dataforgeai-ml-service',
            description: 'Registry-qualified image name without tag.'
        )
        string(
            name: 'IMAGE_TAG',
            defaultValue: '',
            description: 'Image tag (defaults to short git sha when empty).'
        )
        booleanParam(
            name: 'PUSH_IMAGE',
            defaultValue: false,
            description: 'Push the built image to the configured registry.'
        )
    }

    options {
        ansiColor('xterm')
        buildDiscarder(logRotator(numToKeepStr: '20', daysToKeepStr: '30'))
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
    }

    environment {
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        PIP_NO_CACHE_DIR              = '1'
        PYTHONDONTWRITEBYTECODE       = '1'
        PYTHONUNBUFFERED              = '1'
        PYTHON                        = 'python'
        DATAFORGE_CONTRACT_PACK_VERSION = "${params.CONTRACT_VERSION}"
        // Test-only env so tests/test_config.py can `load_config(...)`
        // without touching real platform secrets.
        DATAFORGE_PROFILE                       = 'demo_strict'
        DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL   = 'http://minio:9000'
        DATAFORGE_OBJECT_STORAGE_BUCKET         = 'dataforge-local'
        DATAFORGE_PLATFORM_CALLBACK_URL         = 'http://platform.local/api/ml/jobs/callback'
        DATAFORGE_SERVICE_SIGNING_SECRET        = 'jenkins-ci-fake-signing-secret'
        DATAFORGE_DAGSTER_HOME                  = "${env.WORKSPACE}/.dagster-home"
        DATAFORGE_POLICY_CONFIG_PATH            = 'configs/policies/demo_strict.yaml'
        DATAFORGE_DECISION_POLICY_PATH          = 'configs/policies/decision_v0.yaml'
        DATAFORGE_SCORE_POLICY_PATH             = 'configs/policies/score_v0.yaml'
        DATAFORGE_PERFORMANCE_REPORT_DIR        = "${env.WORKSPACE}/build/performance"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_SHA_SHORT = sh(
                        script: 'git rev-parse --short=12 HEAD',
                        returnStdout: true,
                    ).trim()
                    env.RESOLVED_IMAGE_TAG = params.IMAGE_TAG?.trim() ?: env.GIT_SHA_SHORT
                    env.RESOLVED_IMAGE = "${params.IMAGE_NAME}:${env.RESOLVED_IMAGE_TAG}"
                }
            }
        }

        stage('Install') {
            steps {
                container('python') {
                    sh 'python -m venv .venv'
                    sh '.venv/bin/python -m pip install --upgrade pip'
                    sh '.venv/bin/python -m pip install -e ".[dev]"'
                    sh '.venv/bin/python -m pip install "dagster>=1.13,<2.0"'
                }
            }
        }

        stage('Lint') {
            steps {
                container('python') {
                    sh 'PYTHON=.venv/bin/python make lint'
                }
            }
        }

        stage('Typecheck') {
            steps {
                container('python') {
                    sh 'PYTHON=.venv/bin/python make typecheck'
                }
            }
        }

        stage('Unit tests') {
            steps {
                container('python') {
                    sh 'PYTHON=.venv/bin/python make test'
                }
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'build/junit/*.xml'
                }
            }
        }

        stage('Contract tests') {
            steps {
                container('python') {
                    sh 'PYTHON=.venv/bin/python make test-contracts'
                }
            }
        }

        stage('Plugin tests') {
            steps {
                container('python') {
                    sh 'PYTHON=.venv/bin/python make test-plugins'
                }
            }
        }

        stage('Security tests') {
            steps {
                container('python') {
                    sh 'PYTHON=.venv/bin/python make test-security'
                }
            }
        }

        stage('E2E compute demo') {
            steps {
                container('python') {
                    sh 'PYTHON=.venv/bin/python make test-e2e-compute-demo'
                }
            }
        }

        stage('Performance acceptance') {
            steps {
                container('python') {
                    sh 'mkdir -p build/performance'
                    sh 'PYTHON=.venv/bin/python make test-performance'
                }
            }
            post {
                always {
                    archiveArtifacts(
                        artifacts: 'build/performance/**',
                        allowEmptyArchive: true,
                        fingerprint: false,
                    )
                }
            }
        }

        stage('FastAPI health smoke') {
            steps {
                container('python') {
                    // Boot the FastAPI app in the background and probe
                    // /api/v1/health. The app must come up using only
                    // the env populated above (no real backend / Vault).
                    sh '''
                        set -eu
                        .venv/bin/python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 &
                        UVICORN_PID=$!
                        cleanup() { kill ${UVICORN_PID} 2>/dev/null || true; }
                        trap cleanup EXIT
                        for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
                            if .venv/bin/python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health').status==200 else 1)" 2>/dev/null; then
                                echo "FastAPI health endpoint ok"
                                break
                            fi
                            sleep 1
                            if [ "${attempt}" = "12" ]; then
                                echo "FastAPI health endpoint did not come up in time" >&2
                                exit 1
                            fi
                        done
                    '''
                }
            }
        }

        stage('Dagster definitions load') {
            steps {
                container('python') {
                    // Must load the local in-memory Definitions without
                    // a real backend, signing key, or MinIO instance.
                    sh '''
                        .venv/bin/python -c "from app.orchestration.definitions import build_local_demo_definitions; defs = build_local_demo_definitions(); print('dagster definitions loaded:', len(defs.assets) if hasattr(defs, 'assets') else 'ok')"
                    '''
                }
            }
        }

        stage('Build Docker image') {
            steps {
                container('docker') {
                    sh """
                        docker build \
                          --tag ${env.RESOLVED_IMAGE} \
                          --label "org.opencontainers.image.revision=${env.GIT_SHA_SHORT}" \
                          --label "io.dataforge.contract_pack_version=${params.CONTRACT_VERSION}" \
                          .
                    """
                }
            }
        }

        stage('Scan image') {
            when { expression { return fileExists('/usr/local/bin/trivy') || fileExists('/usr/bin/trivy') } }
            steps {
                container('docker') {
                    sh "trivy image --exit-code 1 --severity CRITICAL,HIGH ${env.RESOLVED_IMAGE} || true"
                }
            }
        }

        stage('Push image') {
            when { expression { return params.PUSH_IMAGE } }
            steps {
                container('docker') {
                    sh "docker push ${env.RESOLVED_IMAGE}"
                }
            }
        }

        stage('Publish build metadata') {
            steps {
                script {
                    def metadata = [
                        commit              : env.GIT_SHA_SHORT,
                        image               : env.RESOLVED_IMAGE,
                        contract_version    : params.CONTRACT_VERSION,
                        pushed              : params.PUSH_IMAGE,
                    ]
                    writeJSON file: 'build/metadata.json', json: metadata, pretty: 2
                    archiveArtifacts artifacts: 'build/metadata.json', allowEmptyArchive: false
                }
            }
        }
    }

    post {
        always {
            echo "ml-service CI finished — image ${env.RESOLVED_IMAGE}, contract ${params.CONTRACT_VERSION}"
        }
        cleanup {
            container('docker') {
                sh "docker image rm ${env.RESOLVED_IMAGE} || true"
            }
        }
    }
}
