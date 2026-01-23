package org.bvd.appendix

import org.junit.jupiter.api.Test

class DataSourceTest {

    @Test
    fun test() {
        Configuration().getDataSource().connection.use { connection ->
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

}