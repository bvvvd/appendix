package org.bvd.appendix

import com.zaxxer.hikari.HikariConfig
import com.zaxxer.hikari.HikariDataSource
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import javax.sql.DataSource

class DataSourceTest {

    @Test
    fun test() {
        getDataSource().connection.use { connection ->
            connection.autoCommit = false

            connection.createStatement().use { st ->
                st.execute("create schema if not exists appendix")
                st.execute("set search_path to appendix")
            }

            connection.prepareStatement(
                """create table if not exists test(id serial primary key, name varchar(255) not null)"""
            ).use { it.execute() }

            connection.prepareStatement("""insert into test(name) values (?)""").use { ps ->
                ps.setString(1, "example name")
                ps.executeUpdate()
            }

            connection.prepareStatement("""select id, name from test""").use { ps ->
                ps.executeQuery().use { rs ->
                    while (rs.next()) println("id: ${rs.getInt("id")}, name: ${rs.getString("name")}")
                }
            }

            connection.commit()
        }
    }

    @Test
    fun testAppendixInstantiation() {
        val appendix = Appendix(getDataSource())
        assertThat(appendix).isNotNull
    }

    fun getDataSource(): DataSource {
        val config = HikariConfig().apply {
            jdbcUrl = "jdbc:postgresql://localhost:5432/appendix?currentSchema=appendix"
            username = "appendix"
            password = "appendix_pass"

            maximumPoolSize = 10
            minimumIdle = 2
            connectionTimeout = 30_000
            idleTimeout = 600_000
            maxLifetime = 1_800_000
        }

        return HikariDataSource(config)
    }
}