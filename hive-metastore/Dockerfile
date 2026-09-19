FROM apache/hive:3.1.3
USER root
RUN if command -v apt-get > /dev/null; then \
      apt-get update && apt-get install -y --no-install-recommends wget netcat-openbsd && rm -rf /var/lib/apt/lists/*; \
    elif command -v microdnf > /dev/null; then \
      microdnf install wget nmap-ncat && microdnf clean all; \
    fi \
    && wget -q https://jdbc.postgresql.org/download/postgresql-42.7.3.jar -O /opt/hive/lib/postgresql-42.7.3.jar \
    && wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.1.0/hadoop-aws-3.1.0.jar -O /opt/hive/lib/hadoop-aws-3.1.0.jar \
    && wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.271/aws-java-sdk-bundle-1.11.271.jar -O /opt/hive/lib/aws-java-sdk-bundle-1.11.271.jar
COPY conf/metastore-site.xml /opt/hive/conf/hive-site.xml
USER hive
